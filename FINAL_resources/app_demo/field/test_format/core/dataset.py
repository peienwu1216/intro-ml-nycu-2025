import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import random

# ==========================================
# [Control Variable] ImageNet Stats
# ==========================================
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def detect_saliency(img, scale=6, q_value=0.95, target_size=(384, 384)):
    """
    [Control Variable] Saliency Detection
    Uses Spectral Residual approach. Do NOT modify if you want to use the same saliency input.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    W, H = img_gray.shape
    
    # Resize for FFT efficiency
    img_resize = cv2.resize(img_gray, (H // scale, W // scale), interpolation=cv2.INTER_AREA)

    myFFT = np.fft.fft2(img_resize)
    myPhase = np.angle(myFFT)
    myLogAmplitude = np.log(np.abs(myFFT) + 0.000001)
    myAvg = cv2.blur(myLogAmplitude, (3, 3))
    mySpectralResidual = myLogAmplitude - myAvg

    m = np.exp(mySpectralResidual) * (np.cos(myPhase) + complex(1j) * np.sin(myPhase))
    saliencyMap = np.abs(np.fft.ifft2(m)) ** 2
    saliencyMap = cv2.GaussianBlur(saliencyMap, (9, 9), 2.5)
    
    # Resize to target size
    saliencyMap = cv2.resize(saliencyMap, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Thresholding and Normalization
    threshold = np.quantile(saliencyMap.reshape(-1), q_value)
    if threshold > 0:
        saliencyMap[saliencyMap > threshold] = threshold
        saliencyMap = (saliencyMap - saliencyMap.min()) / threshold
        
    return saliencyMap

class CADBDataset(Dataset):
    """
    [Control Variable] Dataset Loader
    Ensures all models are trained/tested on the exact same split and data.
    """
    def __init__(self, split, cfg):
        self.data_path = cfg.DATASET_PATH
        self.image_path = os.path.join(self.data_path, 'images')
        self.score_path = os.path.join(self.data_path, 'composition_scores.json')
        self.split_path = os.path.join(self.data_path, 'split.json')
        self.attr_path = os.path.join(self.data_path, 'composition_attributes.json')
        self.weight_path = os.path.join(self.data_path, 'emdloss_weight.json')
        
        self.split = split
        self.attr_types = cfg.ATTRIBUTE_TYPES
        self.image_size = cfg.IMAGE_SIZE
        
        # Load JSONs
        with open(self.split_path, 'r') as f:
            self.image_list = json.load(f)[split]
        with open(self.score_path, 'r') as f:
            self.comp_scores = json.load(f)
        with open(self.attr_path, 'r') as f:
            self.comp_attrs = json.load(f)
            
        if self.split == 'train':
            with open(self.weight_path, 'r') as f:
                self.image_weight = json.load(f)
        else:
            self.image_weight = None
            
        # [Independent Variable] Augmentation
        # You CAN modify this if you want to test different augmentation strategies,
        # but keep the basic Resize/Normalize for fair comparison.
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
            ])

    def __len__(self):
        return len(self.image_list)
        
    def get_attribute(self, image_name):
        attrs = []
        for attr in self.attr_types:
            val = self.comp_attrs[image_name].get(attr, 0.0)
            attrs.append(val)
        return attrs

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_path, image_name)
        
        # Load Image
        try:
            src = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
            src = Image.new('RGB', (self.image_size, self.image_size))
            
        # [Independent Variable] Random Horizontal Flip
        # Applied to both image and saliency logic
        if self.split == 'train' and random.random() < 0.5:
            src = src.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Transform Image
        im = self.transform(src)
        
        # Scores
        score_mean = self.comp_scores[image_name]['mean']
        score_mean = torch.tensor([score_mean], dtype=torch.float32)
        
        score_dist = self.comp_scores[image_name]['dist']
        score_dist = torch.tensor(score_dist, dtype=torch.float32)
        
        # Attributes
        attrs = torch.tensor(self.get_attribute(image_name), dtype=torch.float32)
        
        # Saliency Map
        src_np = np.asarray(src).copy()
        sal_map = detect_saliency(src_np, target_size=(self.image_size, self.image_size))
        sal_map = torch.from_numpy(sal_map.astype(np.float32)).unsqueeze(0) # [1, H, W]
        
        if self.split == 'train':
            emd_weight = torch.tensor(self.image_weight.get(image_name, 1.0), dtype=torch.float32)
            return im, score_mean, score_dist, sal_map, attrs, emd_weight
        else:
            return im, score_mean, score_dist, sal_map, attrs
