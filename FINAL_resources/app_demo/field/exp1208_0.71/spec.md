這份技術規格書（Technical Specification）是基於我們之前的討論，針對 **CADB 資料集（約 9,500 張圖片）** 的特性，將模型重構為 **CNV2-SAMP (ConvNeXt V2 - Saliency Augmented Multi-pattern)** 架構。

[cite_start]此規格書旨在指導工程師實作一個比原始 SAMP-Net [cite: 1, 16] 更強大、且比純 Swin Transformer 更穩定的構圖評估模型。

---

# Technical Specification: CNV2-SAMP Architecture

**Project:** Image Composition Assessment Optimization
[cite_start]**Baseline:** SAMP-Net (ResNet18) [cite: 14, 75]
**Target Metric:** SRCC > 0.70 (Validation)
[cite_start]**Dataset:** CADB (9,497 images) [cite: 34]

## 1. 系統概述 (System Overview)

[cite_start]本系統旨在利用 **ConvNeXt V2** 的全域響應歸一化 (GRN) 特性與卷積歸納偏置 (Inductive Bias)，解決 Transformer 在小數據集上易過擬合與早期飽和的問題。系統引入 **SGFM (Saliency-Guided Feature Modulation)** 與 **GRN-Aware Attention** 取代原始 SAMP 模組 [cite: 16, 126]，以動態感知構圖佈局。

---

## 2. 系統架構詳細規格 (Architecture Specifications)

### 2.1 輸入層 (Input Layer)
* **Image Input**: $384 \times 384 \times 3$ (RGB)
    * [cite_start]*理由*：構圖評估需要高解析度以捕捉線條與邊緣細節，優於原論文的 $224 \times 224$ [cite: 207]。
* **Saliency Input**: $384 \times 384 \times 1$ (Grayscale)
    * [cite_start]生成方式：Spectral Residual (無監督) [cite: 17, 160]，與原論文一致。

### 2.2 骨幹網路 (Backbone): ConvNeXt V2-Nano
* **Model**: `convnextv2_nano` (Pretrained on ImageNet-1K).
* **Feature Extraction**: 採用多尺度特徵融合 (FPN-like structure)。
    * **Stage 3 Output**: Stride 16, Channel 320 (捕捉局部構圖細節)。
    * **Stage 4 Output**: Stride 32, Channel 640 (捕捉全域佈局結構)。
* **Adaptation**: 使用 $1 \times 1$ Conv + LayerNorm 將 Stage 3 與 Stage 4 的 Channel 統一映射至 **512 (Target Dim)**。
* **Fusion**: 將 Stage 4 上採樣 (Bilinear Interpolate) 至 Stage 3 尺寸並相加。

### 2.3 核心模組 I: SGFM (Saliency-Guided Feature Modulation)
[cite_start]取代原論文的 Flatten+Concat [cite: 164, 169]。利用顯著性圖動態調節特徵圖的統計分佈（類似 Style Transfer 的 AdaIN 機制）。

* **輸入**: Fused Feature $F \in \mathbb{R}^{B \times 512 \times H \times W}$，Saliency $S \in \mathbb{R}^{B \times 1 \times H \times W}$。
* **運算流程**:
    1.  **Scale Predictor**: $\gamma(S) = \text{Sigmoid}(\text{Conv}(S))$
    2.  **Shift Predictor**: $\beta(S) = \text{Conv}(S)$
    3.  **Modulation**:
        $$F_{out} = F \cdot (1 + \gamma(S)) + \beta(S)$$
* **目的**: 強顯著區域的特徵響應，壓抑背景雜訊，保留空間結構。

### 2.4 核心模組 II: GRN-Aware Attention Pooling
[cite_start]取代原論文的 8 種固定 Pattern Pooling [cite: 148, 156]。利用 GRN 機制讓特徵通道進行競爭，自動篩選最具構圖意義的空間區域。

* **結構**:
    1.  **Spatial Mixing**: Depthwise Conv $3 \times 3$。
    2.  **Feature Competition**: **Global Response Normalization (GRN)** 層。
        * 計算全域 L2 Norm，進行除法歸一化，增強顯著通道，抑制冗餘通道。
    3.  **Attention Map Generation**: Pointwise Conv $\to$ Softmax (Spatial)。
    4.  **Weighted Pooling**: 將 Attention Map 與特徵圖進行加權總和。

---

## 3. 損失函數規格 (Loss Function Specification)

為了優化排序能力 (SRCC)，採用複合損失函數：
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{wEMD} + \lambda_2 \mathcal{L}_{atts} + \lambda_3 \mathcal{L}_{rank}$$

1.  **Weighted EMD Loss ($\lambda_1 = 1.0$)**:
    * [cite_start]沿用原論文設計 [cite: 17, 192]，消除 Content Bias。
    * [cite_start]$r=2$，使用 Sample-specific weights $\beta$ [cite: 733]。
2.  **Attribute Loss ($\lambda_2 = 0.1$)**:
    * [cite_start]MSE Loss，預測 5 個輔助屬性 (Rule of Thirds, Symmetry 等) [cite: 180, 190]。
3.  **Pairwise Ranking Loss ($\lambda_3 = 0.2$)**: **(新增)**
    * **Type**: MarginRankingLoss。
    * **Logic**: 若 Ground Truth $y_i > y_j$，則預測值需滿足 $\hat{y}_i > \hat{y}_j + \text{margin}$。
    * **Margin**: 設定為 $0.05$。

---

## 4. 訓練與優化策略 (Training & Optimization)

針對小數據集 (Small Data) 的穩定性優化：

### 4.1 優化器 (Optimizer)
* **Type**: AdamW
* **Weight Decay**: $1e-4$ (比 Swin 略大，利用 CNN 的權重正則化防止過擬合)。
* **Layer-wise LR Decay**: 不強制，但建議 Head LR ($2e-4$) 略大於 Backbone LR ($1e-5$)。

### 4.2 訓練技巧
* **EMA (Exponential Moving Average)**: **必須實作**。
    * Decay: $0.999$ or $0.9999$.
    * 解決 Epoch 6 早期飽和與震盪問題，平滑 Validation Curve。
* **Data Augmentation**:
    * Resize ($384 \times 384$).
    * Random Horizontal Flip ($p=0.5$).
    * **Color Jitter**: Brightness=0.1, Contrast=0.1 (微幅調整，增加魯棒性)。
    * [cite_start]**禁止**: Random Crop (會破壞構圖完整性) [cite: 22]。

---

## 5. 驗收標準 (Acceptance Criteria)

工程師需提交包含以下指標的驗證報告：

| 指標 (Metric) | 目標值 (Target) | 說明 |
| :--- | :--- | :--- |
| **SRCC** | **> 0.70** | 斯皮爾曼等級相關係數 (最關鍵指標) |
| **MSE** | **< 0.36** | [cite_start]均方誤差 (需低於論文的 0.3867 [cite: 218]) |
| **Inference** | < 40ms | 單張推理速度 (T4 GPU) |
| **Stability** | Epoch > 10 | 驗證曲線需在 Epoch 10 後仍保持平穩或上升 (由 EMA 保證) |

## 6. 交付物清單 (Deliverables)
1.  `model_cnv2.py`: 包含 GRN Layer, SGFM Module, 與 CNV2SAMPNet 的實作。
2.  `train_ema.py`: 整合了 Ranking Loss 與 EMA 機制的訓練腳本。
3.  `config_cnv2.py`: 對應的超參數設定檔。
4.  **Grad-CAM Visualization**: 針對新架構的 Attention Map 可視化圖，證明模型關注的是構圖熱點而非雜訊。