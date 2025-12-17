import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# Import from core and models
from config import Config
from core.dataset import CADBDataset
from core.loss import EMDLoss, RankLoss, AttributeLoss
from core.utils import calculate_accuracy, calculate_lcc, calculate_spearmanr, dist2ave, visualize_grad_cam
from models.convnext_v2 import ConvNeXtV2SAMPNet # Import your model here

def train_one_epoch(model, ema_model, loader, optimizer, emd_loss_fn, rank_loss_fn, attr_loss_fn, device, cfg):
    """
    [Control Variable] Training Loop
    Standard training loop with EMA update.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, score_mean, score_dist, saliency, attrs, emd_weights in pbar:
        images = images.to(device)
        score_mean = score_mean.to(device).view(-1)
        score_dist = score_dist.to(device)
        saliency = saliency.to(device)
        attrs = attrs.to(device)
        emd_weights = emd_weights.to(device)
        
        optimizer.zero_grad()
        
        pred_dist, pred_attrs = model(images, saliency)
        pred_mean = dist2ave(pred_dist)
        
        # Losses
        l_emd = emd_loss_fn(score_dist, pred_dist, emd_weights)
        l_rank = rank_loss_fn(pred_mean, score_mean)
        l_attr = attr_loss_fn(pred_attrs, attrs)
        
        total_loss = cfg.LAMBDA_EMD * l_emd + cfg.LAMBDA_RANK * l_rank + cfg.LAMBDA_ATTR * l_attr
        
        total_loss.backward()
        optimizer.step()
        
        # Update EMA
        if ema_model is not None:
            ema_model.update_parameters(model)
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': total_loss.item(), 'emd': l_emd.item(), 'rank': l_rank.item()})
        
    return running_loss / len(loader)

def validate(model, loader, device, emd_loss_r1, emd_loss_r2):
    """
    [Control Variable] Validation Loop
    Calculates metrics exactly as defined in SAMPNet.
    """
    model.eval()
    
    emd_r1_sum = 0.0
    emd_r2_sum = 0.0
    correct_count = 0
    total_count = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, score_mean, score_dist, saliency, attrs in tqdm(loader, desc="Validation"):
            images = images.to(device)
            score_mean = score_mean.to(device).view(-1)
            score_dist = score_dist.to(device)
            saliency = saliency.to(device)
            
            pred_dist, _ = model(images, saliency)
            pred_mean = dist2ave(pred_dist)
            
            # Metrics
            emd_r1_sum += emd_loss_r1(score_dist, pred_dist).item()
            emd_r2_sum += emd_loss_r2(score_dist, pred_dist).item()
            
            c, _ = calculate_accuracy(pred_mean, score_mean)
            correct_count += c
            total_count += images.size(0)
            
            all_preds.append(pred_mean)
            all_targets.append(score_mean)
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    mse = torch.nn.MSELoss()(all_preds, all_targets).item()
    srcc = calculate_spearmanr(all_targets, all_preds)
    lcc = calculate_lcc(all_targets, all_preds)
    acc = correct_count / total_count
    avg_emd_r1 = emd_r1_sum / total_count 
    avg_emd_r2 = emd_r2_sum / total_count
    
    return acc, avg_emd_r1, avg_emd_r2, mse, srcc, lcc

def main():
    cfg = Config()
    cfg.create_dirs()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = CADBDataset('train', cfg)
    test_dataset = CADBDataset('test', cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # [Independent Variable] Model Initialization
    # Change this line to use your new model
    model = ConvNeXtV2SAMPNet(cfg).to(device)
    
    # EMA Model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    
    # Optimizer
    # You can adjust parameter groups here if your model has different needs
    backbone_params = list(map(id, model.backbone.parameters()))
    base_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.LR_BACKBONE}, 
        {'params': base_params, 'lr': cfg.LR_HEAD}                  
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
    
    # Losses
    emd_loss_train = EMDLoss(r=2, reduction='mean')
    emd_loss_r1 = EMDLoss(r=1, reduction='sum') 
    emd_loss_r2 = EMDLoss(r=2, reduction='sum') 
    rank_loss = RankLoss(margin=0.05)
    attr_loss = AttributeLoss()
    
    best_srcc = -1.0
    
    for epoch in range(cfg.MAX_EPOCH):
        print(f"\nEpoch {epoch+1}/{cfg.MAX_EPOCH}")
        
        train_loss = train_one_epoch(model, ema_model, train_loader, optimizer, emd_loss_train, rank_loss, attr_loss, device, cfg)
        
        # Validate using EMA model
        acc, emd1, emd2, mse, srcc, lcc = validate(ema_model, test_loader, device, emd_loss_r1, emd_loss_r2)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Results (EMA): Acc={acc:.2%}, EMD1={emd1:.4f}, EMD2={emd2:.4f}, MSE={mse:.4f}, SRCC={srcc:.4f}, LCC={lcc:.4f}")
        
        scheduler.step(train_loss) 
        
        # Save Best
        if srcc > best_srcc:
            best_srcc = srcc
            save_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(ema_model.module.state_dict(), save_path) 
            print(f"Saved best model to {save_path}")
            
            # Generate Grad-CAM for a sample
            try:
                sample_img, _, _, sample_sal, _ = test_dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device)
                sample_sal = sample_sal.unsqueeze(0).to(device)
                cam_path = os.path.join(cfg.LOG_DIR, f'gradcam_epoch_{epoch+1}.png')
                visualize_grad_cam(model, sample_img, sample_sal, cam_path)
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")

if __name__ == '__main__':
    main()
