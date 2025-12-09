import torch
from torchvision import transforms
from config import Config
from dataset import AADBDataset
import sys

def test_dataset():
    print("Testing dataset loading...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        print(f"Label Dir: {Config.LABEL_DIR}")
        print(f"Image Dir: {Config.IMAGE_DIR}")
        
        dataset = AADBDataset(label_dir=Config.LABEL_DIR, img_dir=Config.IMAGE_DIR, split='Validation', transform=transform)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            img, targets = dataset[0]
            print(f"Sample 0 Image shape: {img.shape}")
            print(f"Sample 0 Targets: {targets}")
            print("Targets format: [Score, C, L, F, P, O]")
            print("Dataset read successfully!")
        else:
            print("Dataset is empty!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()

