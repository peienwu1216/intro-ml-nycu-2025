# AADB 美學多輸出模型

## [模型架構]
- Backbone: ResNet18（ImageNet pretrained）
- Shared head: Linear 512→256 + ReLU + Dropout
- 輸出 6 個分數（0–10）：
  - IAS：整體美感
  - C / L / F / P / O：構圖、光線、清晰度、後製、故事感

## [主要檔案]
- ResNet-modified.ipynb: AADBIASDataset，讀取 CSV + 圖片
- model.py: AestheticMultiHeadNet 模型
- train.py: 訓練 + 驗證 + 存最佳模型
- bestmodel.pth: 訓練後最佳權重（state_dict）

## [如何訓練（Colab）]
1. 掛載 Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. 確認路徑（CSV + 圖片資料夾）有改成自己的。

3. 依序跑：
   - Dataset / Transform 定義
   - 建立 train_loader、val_loader
   - 訓練迴圈
4. 程式會自動存：
   - aadb_resnet18_multitask_best.pt
   - bestmodel.pth

## [小報告]
經過修改，包括
  - Loss計算改成 IAS 用 SmoothL1、五個屬性用 MSE，且 IAS 權重比較大（1.5×）
  - train 用隨機裁切、翻轉、輕微色彩抖動；val / 推論用乾淨的 Resize+Normalize
  - 改用 AdamW、更多 epoch、加 Cosine LR scheduler

模型表現有提升到
Pairwise accuracy (IAS_good > IAS_bad): 88.9%
Average IAS margin (good - bad): 1.801

用我們的照片測試
```text
     set  good_IAS   bad_IAS    margin  correct
0   set1  7.086740  2.787169  4.299571        1
1  set10  5.891729  6.714184 -0.822455        0
2   set2  6.715025  6.275054  0.439971        1
3   set3  6.886931  3.570341  3.316590        1
4   set4  5.969162  4.629200  1.339962        1
5   set5  7.011788  4.890609  2.121179        1
6   set6  6.451916  5.184603  1.267313        1
7   set7  6.120734  4.856381  1.264353        1
8   set8  7.899575  4.920441  2.979135        1
```
