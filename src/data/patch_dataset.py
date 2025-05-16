# Author: Brandon Henley

import json
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PatchPairDataset(Dataset):
    def __init__(self, pairs_json_path, patch_size=64, num_pairs=10, transform=None, offset_range=5, include_negatives=True):
        with open(pairs_json_path, 'r') as f:
            self.pairs = json.load(f)

        self.patch_size = patch_size
        self.num_pairs = num_pairs  # number of samples per image pair per epoch
        self.offset_range = offset_range
        self.include_negatives = include_negatives
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs) * self.num_pairs

    def __getitem__(self, idx):
        pair_idx = idx // self.num_pairs
        pair = self.pairs[pair_idx]

        img1 = self.transform(Image.open(pair['img1']).convert('RGB'))
        img2 = self.transform(Image.open(pair['img2']).convert('RGB'))

        c, h, w = img1.shape
        ph = self.patch_size

        # Random crop location
        x = random.randint(0, w - ph - self.offset_range)
        y = random.randint(0, h - ph - self.offset_range)

        dx = random.randint(-self.offset_range, self.offset_range)
        dy = random.randint(-self.offset_range, self.offset_range)

        x2 = min(max(x + dx, 0), w - ph)
        y2 = min(max(y + dy, 0), h - ph)

        patch1 = img1[:, y:y+ph, x:x+ph]

        if self.include_negatives and random.random() < 0.5:
            # Sample negative patch from a different image pair
            neg_idx = random.randint(0, len(self.pairs) - 1)
            while neg_idx == pair_idx:
                neg_idx = random.randint(0, len(self.pairs) - 1)
            neg_pair = self.pairs[neg_idx]
            neg_img = self.transform(Image.open(neg_pair['img2']).convert('RGB'))
            patch2 = neg_img[:, y2:y2+ph, x2:x2+ph]
            label = -1.0
        else:
            patch2 = img2[:, y2:y2+ph, x2:x2+ph]
            label = 1.0

        return patch1, patch2, torch.tensor(label, dtype=torch.float32)
