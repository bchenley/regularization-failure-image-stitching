# Author: Brandon Henley

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from pathlib import Path
from tqdm import tqdm

from src.models.cnn_descriptor import CNNDescriptor
from src.data.patch_dataset import PatchPairDataset

def train_descriptor(    
    pairs_json_path,
    model = None,
    num_epochs=10,
    batch_size=64,
    patch_size=64,
    offset_range=5,
    learning_rate=1e-3,
    weight_decay=0.0,
    use_dropout=False,
    dropout_rate=0.3,
    margin=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_workers=None,
    val_split=0.1
):  
    print(f"Training descriptor (dropout={use_dropout}, L2={weight_decay > 0})")

    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)

    # Full dataset
    full_dataset = PatchPairDataset(
        pairs_json_path, 
        patch_size=patch_size, 
        offset_range=offset_range, 
        include_negatives=True
    )

    # Train/val split
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )

    # Model
    if model is None:
        model = CNNDescriptor().to(device)
    
    model = model.to(device)
    
    print(f"Training on device: {device}")

    # Loss and optimizer
    criterion = nn.CosineEmbeddingLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for patch1, patch2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            patch1 = patch1.to(device, non_blocking=True)
            patch2 = patch2.to(device, non_blocking=True)
            label = label.to(device)

            desc1 = model(patch1)
            desc2 = model(patch2)

            loss = criterion(desc1, desc2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for patch1, patch2, label in val_loader:
                patch1 = patch1.to(device, non_blocking=True)
                patch2 = patch2.to(device, non_blocking=True)
                label = label.to(device)
                desc1 = model(patch1)
                desc2 = model(patch2)
                loss = criterion(desc1, desc2, label)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    def summarize_metrics():
        return {
            "Final Train Loss": round(train_loss_history[-1], 4),
            "Final Val Loss": round(val_loss_history[-1], 4),
            "Epochs Trained": num_epochs,
            "Patch Size": f"{patch_size}x{patch_size}",
            "Offset Range": f"±{offset_range} px",
            "Batch Size": batch_size,
            "Optimizer": "Adam",
            "Use Dropout": use_dropout,
            "Dropout Rate": dropout_rate if use_dropout else 0.0,
            "Weight Decay": weight_decay,
            "Device": device
        }

    return model, train_loss_history, val_loss_history, summarize_metrics()

def save_model(model, path=None):
    if path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = f"outputs/models/descriptor_{timestamp}.pt"
    os.makedirs(Path(path).parent, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(path, input_shape=(3, 64, 64), use_dropout=False, dropout_rate=0.3, device="cpu"):
    model = CNNDescriptor(input_shape=input_shape,
                          use_dropout=use_dropout,
                          dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from: {path}")
    return model


def evaluate_descriptor(model, dataset, device="cpu", num_batches=5, num_workers=None):

    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    model.eval()
    total_cos_sim = 0.0
    count = 0

    with torch.no_grad():
        for i, (patch1, patch2) in enumerate(loader):
            if i >= num_batches:
                break
            patch1 = patch1.to(device, non_blocking=True)
            patch2 = patch2.to(device, non_blocking=True)
            desc1 = model(patch1)
            desc2 = model(patch2)
            cos_sim = cosine_similarity(desc1, desc2).mean().item()
            total_cos_sim += cos_sim
            count += 1

    avg_sim = total_cos_sim / count
    print(f"Average cosine similarity over {count} batches: {avg_sim:.4f}")
    return avg_sim