# main.py

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import sys

# 1. Baseline CNN (Standard)
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # Standard Conv -> ReLU -> MaxPool -> ... -> Linear
        # Constraints: NO BN, NO Dropout, NO Residual
        
        # Block 1: 1x28x28 -> 32x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 32x14x14 -> 64x7x7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten & Fully Connected
        # Input: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 1b. Baseline NN (MLP)
class BaselineNN(nn.Module):
    def __init__(self):
        super(BaselineNN, self).__init__()
        # Requirement: Linear -> ReLU -> Linear -> ReLU -> Linear
        # Input: 784 (28x28 flattened)
        # Hidden sizes: Arbitrary choice for baseline, e.g., 512 -> 256
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # x shape: (B, 784) expected. 
        # Flatten just in case input is (B, 1, 28, 28)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 2. Improved NN (MLP with BN / Dropout)
class ImprovedNN(nn.Module):
    def __init__(self, use_bn=True, use_dropout=True):
        super(ImprovedNN, self).__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        
        # Design: Deeper/Wider MLP 
        # 784 -> 1024 -> 512 -> 256 -> 10
        
        layers = []
        
        # Block 1
        layers.append(nn.Linear(784, 1024))
        if self.use_bn:
            layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.ReLU())
        if self.use_dropout:
            layers.append(nn.Dropout(0.3))
            
        # Block 2
        layers.append(nn.Linear(1024, 512))
        if self.use_bn:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU())
        if self.use_dropout:
            layers.append(nn.Dropout(0.3))
            
        # Block 3
        layers.append(nn.Linear(512, 256))
        if self.use_bn:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        if self.use_dropout:
            layers.append(nn.Dropout(0.3))
            
        # Output
        layers.append(nn.Linear(256, 10))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# 2b. Improved CNN (Optimized for Efficiency and Accuracy: Depth > Width)
class ImprovedCNN(nn.Module):
    def __init__(self, use_bn=True, use_dropout=True, use_residual=False, pooling_type='max'):
        super(ImprovedCNN, self).__init__()
        
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.pooling_type = pooling_type

        # Philosophy: Depth > Width, with increased channel capacity
        # Structure: 3 Blocks. Each block has 2 layers of 3x3 convs.
        # Channels: 32 -> 64 -> 256 (Increased width for better feature representation)
        # BN used after every Conv. Reduced Dropout (GAP provides regularization).
        
        # Helper to create a block component
        def create_conv_layer(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)]
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # --- Block 1: 1 -> 32 ---
        self.block1_conv1 = create_conv_layer(1, 32)
        self.block1_conv2 = create_conv_layer(32, 32)
        
        if self.use_residual:
            # Input 1 -> Output 32. 1x1 conv needed to match channels.
            self.res1 = nn.Conv2d(1, 32, kernel_size=1, bias=False)
        
        # Pooling 1
        if self.pooling_type == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'stride':
            self.pool1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        else:
            self.pool1 = nn.Identity()

        # --- Block 2: 32 -> 64 ---
        self.block2_conv1 = create_conv_layer(32, 64)
        self.block2_conv2 = create_conv_layer(64, 64)

        if self.use_residual:
            # Input 32 -> Output 64. 1x1 conv needed.
            self.res2 = nn.Conv2d(32, 64, kernel_size=1, bias=False)

        # Pooling 2
        if self.pooling_type == 'max':
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.pooling_type == 'stride':
            self.pool2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        else:
            self.pool2 = nn.Identity()
        
        # --- Block 3: 64 -> 256 ---
        self.block3_conv1 = create_conv_layer(64, 128)
        self.block3_conv2 = create_conv_layer(128, 256)

        if self.use_residual:
            # Input 64 -> Output 256. 1x1 conv needed.
            self.res3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)

        # Output
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        if self.use_dropout:
            # Reduced Dropout: GAP already provides strong regularization
            # 0.2 is more appropriate than 0.5 for GAP-based architectures
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Identity()
            
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        identity = x
        out = self.block1_conv1(x)
        out = self.block1_conv2(out)
        
        if self.use_residual:
            # Resize identity to match output channels
            identity = self.res1(identity)
            out = out + identity
        
        out = self.pool1(out)

        # Block 2
        identity = out
        out = self.block2_conv1(out)
        out = self.block2_conv2(out)
        
        if self.use_residual:
            identity = self.res2(identity)
            out = out + identity

        out = self.pool2(out)

        # Block 3
        identity = out
        out = self.block3_conv1(out)
        out = self.block3_conv2(out)
        
        if self.use_residual:
            # Note: Block 3 has no pooling at the end in original, 
            # but effectively it does not downsample spatial dim here anyway (GAP comes next)
            # We need to match channels 64 -> 256
            identity = self.res3(identity)
            out = out + identity
        
        out = self.gap(out)
        out = out.view(out.size(0), -1) # Flatten (B, 256)
        
        out = self.dropout(out)
        out = self.fc(out)
        return out


# set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(56)

# set device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Successfully enabled Mac MPS acceleration mode!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Successfully enabled NVIDIA CUDA mode")
else:
    DEVICE = torch.device("cpu")
    print("No GPU detected, using CPU mode")

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, mode='train', transform=None, model_type='nn'):
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = transform
        self.model_type = model_type
        # 處理資料讀取邏輯
        if self.mode == 'train':
            self.labels = self.data.iloc[:, 0].values
            self.pixels = self.data.iloc[:, 1:].values
        else:
            # 測試集: id 在第0欄, 像素從第2欄開始 (忽略第1欄 label)
            self.ids = self.data.iloc[:, 0].values
            self.pixels = self.data.iloc[:, 2:].values
        
    def __len__(self):
        return len(self.pixels)
        
    def __getitem__(self, idx):
        pixel_data = self.pixels[idx]
        img_array = pixel_data.reshape(28, 28).astype(np.uint8)
        
        if self.transform:
            # Transform expects PIL Image or numpy array
            img_tensor = self.transform(img_array)
        else:
            img_tensor = torch.from_numpy(img_array.astype('float32') / 255.0)
            img_tensor = img_tensor.unsqueeze(0) # 變成 (1, 28, 28)
        
        if self.model_type == 'nn':
            img_tensor = img_tensor.flatten()
        
        if self.mode == 'train':
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, label
        else:
            return img_tensor, self.ids[idx] # 測試集回傳 ID 方便做表格

# train one epoch
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc="Train", leave=False)
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()   # Initialize gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()  # Compute gradients
        optimizer.step() # Update parameters
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(train_loader)
    acc = 100 * correct / total
    return avg_loss, acc

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval() # Switch to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient computation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions
            
    avg_loss = running_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc

import matplotlib.pyplot as plt # 用來畫圖


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    Also acts as a Model Checkpoint to save the best model to disk.
    """
    def __init__(self, patience=15, verbose=False, delta=0, path='best_checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')  # Initialize with infinity
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        # Monitor validation loss instead of accuracy
        score = -val_loss # Convert loss to score (higher is better for logic consistency)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   [EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'   [Model Saved] Val Loss improved from {self.val_loss_min:.4f} to {val_loss:.4f}')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def start_training(model, train_loader, val_loader, device, epochs=60): # 建議至少 60-80
    # 1. 修正 Loss: 開啟 Label Smoothing (關鍵!)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    # 2. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # 初始 LR 可以稍微調高
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # eta_min 保證最後不會變成 0
    
    # 3. 初始化 Early Stopping (Patience 設為 15，給 Scheduler 足夠時間)
    early_stopping = EarlyStopping(patience=15, verbose=True, path='best_model_fashion.pth')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🚀 Start training (Device: {device})")
    print(f"📋 Config: Epochs={epochs}, Patience={early_stopping.patience}, LabelSmoothing=0.1")
    print("-" * 70)
    
    for epoch in range(epochs):
        # --- Training ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # --- Validation ---
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # --- Scheduler Step (非常重要：通常在每個 Epoch 結束後更新) ---
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- Record History ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # --- Print Status ---
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%", end="")
        
        # --- Early Stopping Check ---
        # 這一步會自動檢查是否是最佳模型，如果是就會存檔
        # Now monitoring validation loss (pass val_loss instead of val_acc)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
            
    print("-" * 70)
    print("🏁 Training loop finished.")
    
    # --- 關鍵：載入訓練過程中表現最好的模型權重 ---
    # 因為最後一個 epoch 不一定是最好的，我們要把最好的那個讀回來
    model.load_state_dict(torch.load('best_model_fashion.pth'))
    print(f"🏆 Loaded best model from 'best_model_fashion.pth' with Loss: {early_stopping.val_loss_min:.4f}")
    
    return history

def generate_predictions(model, test_loader, device, output_file='pred.csv'):
    print("🔮 Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Store (id, prediction)
            for id_val, pred_val in zip(ids, predicted):
                predictions.append((id_val.item(), pred_val.item()))
    
    # Save to CSV
    df = pd.DataFrame(predictions, columns=['idx', 'label'])
    df.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")


def generate_predictions_TTA(model, test_loader, device, output_file='pred_tta.csv'):
    """
    Generate predictions with Test Time Augmentation (TTA).
    """
    print("🔮 Generating predictions with TTA (Test Time Augmentation)...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting (TTA)"):
            inputs = inputs.to(device)
            
            # 1. Forward pass (Original)
            outputs_orig = model(inputs)
            probs_orig = F.softmax(outputs_orig, dim=1)  # 轉成機率
            
            # 2. Forward pass (Flipped) - 模擬水平翻轉
            # dims=[3] 是寬度維度 (W), 對於 (B, C, H, W) 格式
            inputs_flipped = torch.flip(inputs, dims=[3])
            outputs_flipped = model(inputs_flipped)
            probs_flipped = F.softmax(outputs_flipped, dim=1)
            
            # 3. Average Probabilities
            avg_probs = (probs_orig + probs_flipped) / 2.0
            
            # 4. Get Final Prediction
            _, predicted = torch.max(avg_probs, 1)
            
            # Store (id, prediction)
            for id_val, pred_val in zip(ids, predicted):
                predictions.append((id_val.item(), pred_val.item()))
    
    # Save to CSV
    df = pd.DataFrame(predictions, columns=['idx', 'label'])
    df.to_csv(output_file, index=False)
    print(f"✅ TTA Predictions saved to {output_file}")


if __name__ == "__main__":
    # --- 1. Prepare Data with Augmentation ---
    print("Preparing CNN data with augmentation...")
    
    # Training transform with data augmentation (Literature Recommended Pipeline)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28, padding=2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value=0)
    ])
    
    # Validation/Test transform (no augmentation, only normalization)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full dataset (without transform first, for safe splitting)
    full_dataset = FashionMNISTDataset("data/train.csv", mode='train', 
                                       transform=None, model_type='cnn')
    
    # Split Validation
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate datasets
    train_dataset = FashionMNISTDataset("data/train.csv", mode='train', 
                                       transform=train_transform, model_type='cnn')
    val_dataset = FashionMNISTDataset("data/train.csv", mode='train', 
                                      transform=val_transform, model_type='cnn')
    
    train_set = Subset(train_dataset, train_indices.indices)
    val_set = Subset(val_dataset, val_indices.indices)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)
    
    # --- 2. Initialize Models ---
    baseline_model = BaselineCNN().to(DEVICE)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    print(f"📊 Baseline CNN Params: {baseline_params:,}")

    # Updated to use 'stride' based on Ablation Study results
    model = ImprovedCNN(use_bn=True, use_dropout=True, use_residual=True, pooling_type='stride').to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Improved CNN Params: {total_params:,} (Target: Efficient & Accurate)") 
    
    # --- 3. Start Training or Load Model ---
    skip_training = '--skip-training' in sys.argv or '--tta-only' in sys.argv
    
    if skip_training:
        print("\n⏭️  Skipping training, loading existing model...")
        try:
            # Try loading the new best model (if we just trained it)
            model.load_state_dict(torch.load("best_model_fashion.pth", map_location=DEVICE))
            print("✅ Loaded best model from 'best_model_fashion.pth'")
        except Exception as e:
            print(f"❌ Could not load 'best_model_fashion.pth': {e}")
            print("⚠️ Retraining is recommended due to architecture changes.")
            # If we can't load, we fallback to training logic if user allows, 
            # but since user asked for --tta-only, we should probably stop or warn.
            # However, since we just updated the architecture, we MUST retrain.
            # We will force training if model loading fails significantly.
            if '--tta-only' in sys.argv:
                print("⚠️ Cannot perform TTA without a valid trained model matching current architecture.")
                print("⚠️ Please run 'python main.py' first to train the new architecture.")
                sys.exit(1)
    else:
        history = start_training(model, train_loader, val_loader, DEVICE, epochs=60)
    
    # --- 4. Generate Predictions for Kaggle ---
    print("\nLoading Test Data for Kaggle Submission...")
    test_dataset = FashionMNISTDataset("data/test4students.csv", mode='test', 
                                      transform=val_transform, model_type='cnn')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    if '--tta-only' in sys.argv:
        print("\n🎯 Generating TTA predictions only (pred.csv will be preserved)...")
        generate_predictions_TTA(model, test_loader, DEVICE, output_file='pred_tta.csv')
    else:
        # Generate BOTH Standard and TTA predictions by default
        print("\n🔮 Generating Standard Predictions...")
        generate_predictions(model, test_loader, DEVICE, output_file='pred.csv')
        
        print("\n🔮 Generating TTA Predictions...")
        generate_predictions_TTA(model, test_loader, DEVICE, output_file='pred_tta.csv')

