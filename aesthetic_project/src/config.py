import torch

class Config:
    # 資料路徑
    IMAGE_DIR = "../data/AADB/datasetImages_warp256"
    LABEL_DIR = "../data/AADB/imgListFiles_label"
    
    # 訓練參數
    BATCH_SIZE = 32  # M晶片記憶體若不夠可調小為 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    
    # 模型參數
    MODEL_NAME = 'vit_tiny_patch16_224' # 使用 Tiny ViT 確保在 Mac 上跑得快
    NUM_CLASSES = 6 # 1個總分 + 5個子分數
    
    # 硬體加速
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

