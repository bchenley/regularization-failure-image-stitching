# Author: Brandon Henley

import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PairDataset(Dataset):
    def __init__(self, pairs_json_path, transform=None, resize=(256, 256)):
        with open(pairs_json_path, 'r') as f:
            self.pairs = json.load(f)

        self.transform = transform or transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1 = Image.open(pair['img1']).convert('RGB')
        img2 = Image.open(pair['img2']).convert('RGB')
        return self.transform(img1), self.transform(img2)