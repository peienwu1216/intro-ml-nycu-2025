import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

class AADBDataset(Dataset):
    def __init__(self, json_file, root_dir, split='train', transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Split logic: 8.5k Train, 0.5k Val, 1.0k Test
        # We'll just take the first 8500 for train, next 500 for val, rest for test
        # Assuming the data is shuffled or random enough. 
        # The spec says "Use standard partition". 
        # Since we don't have the standard partition file, we'll do a deterministic split.
        
        # Sort by filename to ensure deterministic split
        self.data.sort(key=lambda x: x['filename'])
        
        # Shuffle deterministically
        rng = np.random.RandomState(42)
        rng.shuffle(self.data)
        
        n_total = len(self.data)
        n_train = 8500
        n_val = 500
        # n_test = rest
        
        if split == 'train':
            self.data = self.data[:n_train]
        elif split == 'val':
            self.data = self.data[n_train:n_train+n_val]
        elif split == 'test':
            self.data = self.data[n_train+n_val:]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Use img_path from json if available, otherwise assume it's in root_dir/filename
        if 'img_path' in item:
            img_path = os.path.join(self.root_dir, item['img_path'])
        else:
            img_path = os.path.join(self.root_dir, item['filename'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if failed
            image = Image.new('RGB', (384, 384))
            
        if self.transform:
            image = self.transform(image)
            
        # Targets
        # IAS (Global), C, L, F, O
        targets = {
            'IAS': torch.tensor(item['IAS'], dtype=torch.float32),
            'C': torch.tensor(item['C'], dtype=torch.float32),
            'L': torch.tensor(item['L'], dtype=torch.float32),
            'F': torch.tensor(item['F'], dtype=torch.float32),
            'O': torch.tensor(item['O'], dtype=torch.float32)
        }
        
        return image, targets

class LetterboxPad:
    def __init__(self, size=384, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill
        
    def __call__(self, img):
        # Resize maintaining aspect ratio
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Pad
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return F.pad(img, padding, self.fill)

