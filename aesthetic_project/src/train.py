import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from config import Config
from dataset import AADBDataset
from model import AestheticViT

def train():
    # 1. 準備資料預處理
    # 訓練集增加資料增強 (Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # 先放大一點
        transforms.RandomCrop((224, 224)), # 再隨機裁切
        transforms.RandomHorizontalFlip(p=0.5), # 隨機水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 隨機顏色調整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 驗證集只做 Resize & Normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    model = AestheticViT(model_name=Config.MODEL_NAME).to(Config.DEVICE)
    criterion = nn.MSELoss() # 回歸問題使用均方誤差
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # 4. 訓練迴圈
    loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
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
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_loss_history.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
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
    
    # 畫 Loss 圖
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    save_path_img = os.path.join(figures_dir, "loss_curve.png")
    plt.savefig(save_path_img)
    print(f"Loss curve saved to {save_path_img}")

if __name__ == "__main__":
    train()
