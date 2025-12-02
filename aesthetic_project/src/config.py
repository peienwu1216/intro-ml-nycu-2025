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
    # 改用 Swin Transformer Tiny 版本
    MODEL_NAME = 'swin_tiny_patch4_window7_224' 
    NUM_CLASSES = 5 # 1個總分 + 4個子分數 (移除 Post-processing)
    
    # 硬體加速
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

