# Image Aesthetics Assessment (Swin-MTL)

**是 Swin Transformer ！ 我用了 Swin Transformer！**

## Dataset的家

**AADB (Aesthetics Assessment Database)** 

葛、姐請將資料集放置於專案根目錄下，結構如下：

```
project_root/
├── ImageAesthetics_ECCV2016/  <-- 資料集主資料夾
│   ├── datasetImages_warp256/ <-- 圖片資料夾
│   ├── AADBinfo.mat           <-- 標籤資訊
│   └── ...
├── train.py
├── test.py
├── best_model.pth <-- 從google雲端下載後丟到這邊
└── ...
```

親親若您使用 `step1_process_data.py` 進行預處理，它會生成 `dataset_processed.json`，請確保該檔案也位於根目錄

## installation

### python requirements環境
小弟是建議用個venv

```bash
pip install -r requirements.txt
```
### 訓練好的模型
[google drive](https://drive.google.com/drive/folders/1oWFEEyo1xTBZ5DHHLpV0tYpZEZXVjT_c?usp=sharing) (大家也可以放在這裡！)


## 重要的程式 Usage ：
> 親您快看！

### 用我訓練好的模型測試圖片！
#### 1. `app.py` (Web 版本)
Flask 的版本，可以看到我用的精美前端
- **用法**：
  ```bash
  python app.py
  ```
- 開啟瀏覽器訪問 `http://127.0.0.1:5000`。

#### 2. `test.py` (command line版本)
commandline 的版本，可針對單or多張圖片或整個資料夾進行評分。
- **用法**：
  ```bash
  # 測試單張圖片
  python test.py path/to/image.jpg

  # 測試整個資料夾
  python test.py path/to/image_folder/
  ```
### 火車模型/模型架構/
#### `train.py` (模型訓練)
用於訓練 Swin-MTL 模型。包含訓練迴圈、驗證以及模型權重儲存。
- **用法**：
  ```bash
  python train.py
  ```
- 訓練過程會自動儲存最佳模型至 `best_model.pth`。

#### `inference.py` (進階驗證與視覺化)
用於批次驗證特定資料集（如 Good/Bad 對比組），並生成視覺化圖表（雷達圖、長條圖）。
- **用法**：
  ```bash
  python inference.py
  ```
- 需確保有 `validation/` 資料夾包含測試集。

#### `step1_process_data.py` (資料預處理)
將原始資料集標籤轉換為訓練所需的 JSON 格式。
- **用法**：
  ```bash
  python step1_process_data.py
  ```

## 模型架構 (Model)
- **Backbone**: Swin Transformer (Tiny)
- **Heads**: 
  - IAS (Overall Aesthetics)
  - Composition (C)
  - Lighting (L)
  - Focus (F)
  - Originality (O) (Optional: uses CLIP features)

## 備註
- `best_model.pth`: 是我預訓練好的模型權重檔。
- `model.py`, `dataset.py`, `loss.py`: 是核心模型定義與資料載入程式碼。
