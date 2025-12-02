import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.stats import spearmanr
from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import AADBDataset, LetterboxPad
from model import SwinMTL_NoPost
from loss import TotalLoss

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_rank = 0.0
    running_reg = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        target_dict = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss, loss_dict = criterion(outputs, target_dict)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_rank += loss_dict['rank'].item()
        running_reg += loss_dict['reg'].item()
        
        pbar.set_postfix({'loss': loss.item(), 'rank': loss_dict['rank'].item(), 'reg': loss_dict['reg'].item()})
        
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    preds_ias = []
    targets_ias = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            outputs = model(images)
            
            preds_ias.extend(outputs['IAS'].cpu().numpy().flatten())
            targets_ias.extend(targets['IAS'].numpy().flatten())
            
    srcc, _ = spearmanr(preds_ias, targets_ias)
    return srcc

def validate_custom_sets(model, validation_dir, device, transform):
    model.eval()
    print("\n--- Custom Validation Sets ---")
    correct_pairs = 0
    total_pairs = 0
    
    if not os.path.exists(validation_dir):
        print(f"Validation directory not found: {validation_dir}")
        return

    # Iterate set1 to set10
    sets = sorted([d for d in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, d)) and d.startswith('set')])
    
    with torch.no_grad():
        for set_name in sets:
            set_path = os.path.join(validation_dir, set_name)
            
            # Find good and bad images
            files = os.listdir(set_path)
            good_file = next((f for f in files if 'good' in f.lower()), None)
            bad_file = next((f for f in files if 'bad' in f.lower()), None)
            
            if not good_file or not bad_file:
                print(f"Skipping {set_name}: Missing good or bad image.")
                continue
                
            good_img_path = os.path.join(set_path, good_file)
            bad_img_path = os.path.join(set_path, bad_file)
            
            try:
                good_img = Image.open(good_img_path).convert('RGB')
                bad_img = Image.open(bad_img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading images in {set_name}: {e}")
                continue
                
            if transform:
                good_tensor = transform(good_img).unsqueeze(0).to(device)
                bad_tensor = transform(bad_img).unsqueeze(0).to(device)
            
            outputs_good = model(good_tensor)
            outputs_bad = model(bad_tensor)
            
            out_good_ias = outputs_good['IAS'].item()
            out_bad_ias = outputs_bad['IAS'].item()
            
            is_correct = out_good_ias > out_bad_ias
            result_str = "✓" if is_correct else "✗"
            if is_correct:
                correct_pairs += 1
            total_pairs += 1
            
            print(f"[{set_name}] {result_str}")
            print(f"  Good: IAS={out_good_ias:.4f} | C={outputs_good['C'].item():.4f}, L={outputs_good['L'].item():.4f}, F={outputs_good['F'].item():.4f}, O={outputs_good['O'].item():.4f}")
            print(f"  Bad : IAS={out_bad_ias:.4f} | C={outputs_bad['C'].item():.4f}, L={outputs_bad['L'].item():.4f}, F={outputs_bad['F'].item():.4f}, O={outputs_bad['O'].item():.4f}")
            
    if total_pairs > 0:
        print(f"Accuracy: {correct_pairs}/{total_pairs} ({correct_pairs/total_pairs*100:.1f}%)")
    print("------------------------------\n")

def main():
    # Config
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'ImageAesthetics_ECCV2016')
    JSON_PATH = os.path.join(BASE_DIR, 'dataset_processed.json')
    VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
    
    # Transforms
    transform = transforms.Compose([
        LetterboxPad(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = AADBDataset(JSON_PATH, BASE_DIR, split='train', transform=transform)
    val_dataset = AADBDataset(JSON_PATH, BASE_DIR, split='val', transform=transform)
    test_dataset = AADBDataset(JSON_PATH, BASE_DIR, split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    model = SwinMTL_NoPost().to(DEVICE)
    
    # Loss & Optimizer
    criterion = TotalLoss(lambda1=1.0, lambda2=5.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training Loop
    best_srcc = -1.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_srcc = validate(model, val_loader, DEVICE)
        
        # Custom Validation
        validate_custom_sets(model, VALIDATION_DIR, DEVICE, transform)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val SRCC: {val_srcc:.4f}")
        
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            torch.save(model.state_dict(), os.path.join(BASE_DIR, 'best_model.pth'))
            print("Saved best model!")
            
    print("\nTraining Complete.")
    print(f"Best Val SRCC: {best_srcc:.4f}")
    
    # Test
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_model.pth')))
    test_srcc = validate(model, test_loader, DEVICE)
    print(f"Test SRCC: {test_srcc:.4f}")
    
    # Custom Validation Sets
    validation_dir = os.path.join(BASE_DIR, 'custom_validation_sets')
    validate_custom_sets(model, validation_dir, DEVICE, transform)

if __name__ == '__main__':
    main()
