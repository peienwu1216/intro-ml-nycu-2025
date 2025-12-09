import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

from config import Config
from dataset import AADBDataset
from model import AestheticViT, AestheticSwin

def train():
    # 1. 準備資料預處理
    # 訓練集增加資料增強 (Data Augmentation)
    # Modified for aesthetic scoring: Avoid heavy cropping and rotation to preserve composition
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), # 直接 Resize 到輸入大小，保留構圖
        # transforms.Resize((232, 232)), # 或者先稍微放大
        # transforms.RandomCrop((224, 224)), # 再做輕微裁切 (Optional)
        
        transforms.RandomHorizontalFlip(p=0.5), # 水平翻轉是安全的
        
        # Remove RandomRotation or keep it very small
        # transforms.RandomRotation(5), 
        
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 降低 Jitter 強度
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 驗證集只做 Resize & Normalize
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 2. 載入資料集
    print(f"Loading datasets from {Config.LABEL_DIR}...")
    
    try:
        train_dataset = AADBDataset(label_dir=Config.LABEL_DIR, img_dir=Config.IMAGE_DIR, split='Train', transform=train_transform)
        val_dataset = AADBDataset(label_dir=Config.LABEL_DIR, img_dir=Config.IMAGE_DIR, split='Validation', transform=val_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check if the file paths in config.py are correct and the files exist.")
        return
    
    print(f"Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Train dataset is empty.")
        return

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. 初始化模型與優化器
    print(f"Initializing model {Config.MODEL_NAME} on {Config.DEVICE}...")
    
    # 根據 Config 中的名稱決定呼叫哪個 Class
    if 'swin' in Config.MODEL_NAME:
        model = AestheticSwin(model_name=Config.MODEL_NAME).to(Config.DEVICE)
    else:
        model = AestheticViT(model_name=Config.MODEL_NAME).to(Config.DEVICE)
        
    criterion = nn.MSELoss() # 回歸問題使用均方誤差
    
    # --- Two-Stage Training Strategy ---
    
    # Stage 1: Freeze Backbone, Train Heads Only
    print("Stage 1: Training Heads only (Backbone frozen)")
    model.freeze_backbone()
    
    # Use a larger LR for training heads from scratch
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # Train for 5 epochs
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/5")
        
        for images, targets in loop:
            images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Stage 1 Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

    # Stage 2: Unfreeze Backbone, Fine-tune Entire Model
    print("Stage 2: Fine-tuning entire model")
    model.unfreeze_backbone()
    
    # Use smaller LR and higher weight decay for fine-tuning ViT
    # weight_decay=1e-2 to 5e-2 is recommended for ViT
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 4. 訓練迴圈
    loss_history = []
    val_loss_history = []
    val_mae_history = []   # 新增
    val_srcc_history = []  # 新增
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Starting Stage 2 training...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        for images, targets in loop:
            images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 計算 Loss: 總分 Loss + 5個子分數 Loss
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        loss_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Collect predictions and targets for metrics
                val_preds_list.append(outputs.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_loss_history.append(avg_val_loss)
        
        # Calculate Metrics (MAE & SRCC)
        mae = 0.0
        srcc = 0.0
        if len(val_preds_list) > 0:
            val_preds = np.concatenate(val_preds_list, axis=0)
            val_targets = np.concatenate(val_targets_list, axis=0)
            
            # Index 0 is the Global Aesthetic Score (IAS)
            # MAE: Mean Absolute Error
            mae = np.mean(np.abs(val_preds[:, 0] - val_targets[:, 0]))
            
            # SRCC: Spearman's Rank Correlation Coefficient
            srcc, _ = stats.spearmanr(val_preds[:, 0], val_targets[:, 0])

        val_mae_history.append(mae)
        val_srcc_history.append(srcc)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"          Validation Metrics (IAS) -> MAE: {mae:.4f}, SRCC: {srcc:.4f}")
        
        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping & Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = "aesthetic_vit_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}!")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        
    # 5. 訓練結束
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 圖表 1: Loss & MAE (誤差類，越低越好)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.plot(val_mae_history, label='Val MAE', linestyle='--')
    plt.title("Loss & MAE (Lower is Better)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # 圖表 2: SRCC (相關性，越高越好)
    plt.subplot(1, 2, 2)
    plt.plot(val_srcc_history, label='Val SRCC', color='green')
    plt.title("Spearman Correlation (Higher is Better)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path_img = os.path.join(figures_dir, "metrics_curve.png")
    plt.savefig(save_path_img)
    print(f"📊 Metrics curve saved to {save_path_img}")

if __name__ == "__main__":
    train()
