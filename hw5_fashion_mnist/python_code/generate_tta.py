# generate_tta.py
# 載入已訓練模型並使用 TTA 生成預測

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# 導入模型類和數據集類
from main import ImprovedCNN, FashionMNISTDataset

# 設置設備
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Mac MPS acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using NVIDIA CUDA")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Using CPU")

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
    print("=" * 70)
    print("🚀 TTA Prediction Generator")
    print("=" * 70)
    
    # --- 1. 載入模型 ---
    print("\n📦 Loading model...")
    model = ImprovedCNN().to(DEVICE)
    
    # 優先使用 best_model_fashion.pth（EarlyStopping 保存的最佳模型）
    model_path = "best_model_fashion.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Loaded model from {model_path}")
    except FileNotFoundError:
        # 如果沒有 best_model，嘗試載入 improved_cnn.pth
        model_path = "improved_cnn.pth"
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Loaded model from {model_path}")
    
    # --- 2. 準備測試數據 ---
    print("\n📊 Preparing test data...")
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = FashionMNISTDataset("data/test4students.csv", mode='test', 
                                      transform=val_transform, model_type='cnn')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    # --- 3. 使用 TTA 生成預測 ---
    print("\n" + "=" * 70)
    generate_predictions_TTA(model, test_loader, DEVICE, output_file='pred_tta.csv')
    print("=" * 70)
    print("🎉 Done! pred_tta.csv has been generated.")
    print("💡 Original pred.csv is preserved.")

