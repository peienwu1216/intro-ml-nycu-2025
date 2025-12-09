# Aesthetic Assessment Project - ViT Baseline

期末專案：基於 Vision Transformer 的美學評分系統

## 📁 專案結構

```
aesthetic_project/
├── data/                      # 資料目錄
│   └── AADB/                 # AADB 資料集
│       ├── datasetImages_warp256/    # 圖片資料夾
│       └── imgListFiles_label/       # 標籤檔案 (txt格式)
├── src/                       # 原始碼
│   ├── config.py             # 設定檔（路徑、超參數）
│   ├── dataset.py            # 資料集載入與 Label Mapping
│   ├── model.py              # ViT 模型架構
│   ├── train.py              # 訓練腳本
│   └── test_dataset.py       # 資料集測試腳本
├── venv/                     # 虛擬環境（已在 .gitignore 中）
├── requirements.txt          # Python 套件依賴
└── README.md                 # 本文件
```

## 🚀 快速開始

### 1. 啟動虛擬環境

```bash
cd aesthetic_project
source venv/bin/activate
```

你會看到終端機前面出現 `(venv)` 標示，表示虛擬環境已啟動。

### 2. 安裝依賴（首次執行）

```bash
pip install -r requirements.txt
```

### 3. 測試資料集載入

```bash
cd src
python test_dataset.py
```

應該會看到：
```
Dataset length: 8458
Sample 0 Image shape: torch.Size([3, 224, 224])
Sample 0 Targets: tensor([0.3000, 0.4500, 0.5333, 0.3667, 0.5000, 0.4000])
Dataset read successfully!
```

### 4. 開始訓練

```bash
python train.py
```

訓練過程會顯示：
- 每個 Epoch 的 Train Loss 和 Validation Loss
- 進度條顯示每個 Batch 的 Loss
- 訓練結束後會自動儲存模型和 Loss 曲線圖

## ⚙️ 設定檔說明

編輯 `src/config.py` 可以調整：

- **資料路徑**：`IMAGE_DIR` 和 `LABEL_DIR`
- **訓練參數**：`BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- **模型參數**：`MODEL_NAME`（可選 'vit_tiny_patch16_224', 'vit_small_patch16_224' 等）

## 📊 Label Mapping 邏輯

程式碼會自動將 AADB 的 11 個原始屬性映射為 5 個美學維度：

- **Lighting/Color (L)** = Avg(Good Lighting, Color Harmony, Vivid Color)
- **Composition (C)** = Avg(Rule of Thirds, Balancing Element, Symmetry, Repetition)
- **Focus/Clarity (F)** = Avg(Shallow DOF, Object Emphasis, 1 - Motion Blur)
- **Originality/Story (O)** = Interesting Content
- **Post-processing (P)** = 0.5 (常數，因 AADB 無直接對應標籤)

最終輸出：`[IAS (總分), C, L, F, P, O]` (6 維向量)

## 💻 系統需求

- **作業系統**: macOS (M 系列晶片推薦)
- **Python**: 3.10+
- **硬體加速**: MPS (Metal Performance Shaders) - 自動啟用

## 📝 注意事項

1. **虛擬環境獨立性**：此專案的 `venv/` 位於專案目錄下，與課程其他作業（hw1, hw5 等）完全隔離，不會互相影響。
2. **資料路徑**：如果資料位置不同，請修改 `src/config.py` 中的路徑。
3. **記憶體不足**：如果訓練時出現記憶體錯誤，請將 `BATCH_SIZE` 調小（例如 16 或 8）。

## 🔧 疑難排解

**Q: 找不到資料集檔案？**  
A: 檢查 `src/config.py` 中的 `IMAGE_DIR` 和 `LABEL_DIR` 路徑是否正確。

**Q: 訓練速度太慢？**  
A: 確認終端機顯示 `Using device: mps`，表示 MPS 加速已啟用。也可嘗試將 `BATCH_SIZE` 調大（但要注意記憶體限制）。

**Q: 如何停用虛擬環境？**  
A: 在終端機輸入 `deactivate` 即可。

## 📧 聯絡資訊

如有問題，請參考專案文件或聯繫團隊成員。

