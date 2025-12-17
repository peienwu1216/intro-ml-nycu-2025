[cite_start]論文中確實使用了 **MSE (均方誤差)**、**EMD (Earth Mover's Distance)**、**SRCC (斯皮爾曼等級相關係數)** 和 **LCC (線性相關係數)** 作為評估指標 [cite: 209, 210, 211, 212]。

以下是針對超越 SAMP-Net 的 **優化策略規格書 (Optimization Specification)**，分為四個層次：**Backbone 升級**、**結構動態化**、**顯著性增強** 以及 **損失函數優化**。

---

### 優化策略 1：Backbone 現代化 (Modernizing the Backbone)

**分析瓶頸：**
[cite_start]SAMP-Net 使用的是 **ResNet18** [cite: 75, 206][cite_start]。這是一個非常老舊且輕量級的特徵提取器。雖然作者認為它效率高，但在捕捉細微的紋理與全域上下文 (Global Context) 上，遠不如現代模型。此外，輸入圖像僅為 $224 \times 224$ [cite: 207]，對於評估「構圖」這種需要高解析度細節的任務來說過小。

**優化方案：**
1.  **替換為 Swin Transformer 或 ConvNeXt**：
    * **原理**：構圖評估高度依賴於「物體之間的長距離依賴關係」(例如：左下角的物體與右上角的物體是否平衡)。CNN (ResNet) 的感受野 (Receptive Field) 有限，而 **Vision Transformer (如 Swin-B)** 天生具有捕捉全域關係的能力。
    * **預期效果**：提升 SRCC 與 LCC，因為模型能更好地理解整體佈局。

2.  **增加輸入解析度**：
    * **操作**：將 $224 \times 224$ 提升至 **$384 \times 384$** 或更高。
    * **注意**：這會增加計算量，但對於構圖細節（如線條、邊緣）的捕捉至關重要。

---

### 優化策略 2：從「靜態模式」到「動態感知」 (Dynamic Composition Learning)

**分析瓶頸：**
[cite_start]SAMP 模組使用了 **8 種固定的人工設計模式** (如三分法、對稱等) [cite: 148, 149]。
這有兩個缺點：
1.  **剛性 (Rigid)**：並非所有好照片都符合這 8 種模式。
2.  [cite_start]**硬裁切 (Hard Pooling)**：使用固定的網格進行 Pooling [cite: 157]，如果重要物體剛好跨越兩個網格，特徵會被切斷。

**優化方案：**
1.  **引入可變形卷積 (Deformable Convolution) 或 ROI Align**：
    * 不強制使用固定的 $2 \times 2$ 或 $3 \times 3$ 網格，而是讓網絡學習「關注區域」的偏移量 (Offset)。

2.  **Graph Neural Network (GNN) 構圖建模**：
    * **新架構思路**：
        1.  先用 Object Detector (如 YOLO 或 Faster R-CNN) 抓出圖中前 $N$ 個顯著物體。
        2.  將每個物體視為 Graph 的一個 **Node**。
        3.  物體之間的距離、角度、大小比率視為 **Edge**。
        4.  透過 GCN (Graph Convolutional Network) 學習這些節點之間的關係。
    * **優勢**：這比 SAMP 的固定網格更能直接模擬「物體間的平衡與關係」。

---

### 優化策略 3：強化顯著性融合 (Advanced Saliency Fusion)

**分析瓶頸：**
[cite_start]原論文將 Saliency Map 下採樣後直接 **Flatten (拉平)** 成向量，然後與視覺特徵拼接 [cite: 164, 169]。
* **問題**：Flatten 操作破壞了顯著性圖的二維空間結構，模型失去了「顯著物體在哪個位置」的精確空間對應關係。

**優化方案：**
1.  **空間注意力機制 (Spatial Attention)**：
    * 不要 Flatten。將 Saliency Map 作為一個 **Attention Mask** (單通道)，乘回 Backbone 的 Feature Map ($F$)。
    * 公式概念：$F_{new} = F \otimes \text{Sigmoid}(\text{Conv}(Saliency))$
    * 這能讓模型在提取特徵時，自動「聚焦」在顯著區域，而不是後期才硬接合。

2.  **使用深度學習 Saliency Detector**：
    * [cite_start]論文使用的是無監督的 Spectral Residual 方法 [cite: 160][cite_start]，且聲稱當時嘗試的監督式方法效果不佳 [cite: 161]。
    * **反駁/改進**：現在已有更強的 Saliency 模型 (如 **U2Net**, **TRACER**)。使用 SOTA 的顯著性檢測器可以提供更精確的物體邊界，減少背景雜訊干擾。

---

### 優化策略 4：損失函數的進化 (Loss Function Engineering)

**分析瓶頸：**
[cite_start]目前的 Loss 是 $\mathcal{L}_{wEMD} + \lambda \mathcal{L}_{atts}$ [cite: 201]。
* **EMD/MSE** 只能讓預測分數「接近」真實分數。
* [cite_start]**SRCC (排名相關性)** 衡量的是「排序能力」(即 Image A 比 Image B 好，預測分數是否也反映這點)。目前的 Loss 並沒有直接優化 SRCC [cite: 212]。

**優化方案：**
1.  **加入 Pairwise Ranking Loss (成對排名損失)**：
    * 在訓練時，一次輸入兩張圖片 $(I_1, I_2)$，若 Ground Truth $y_1 > y_2$，則強制模型預測 $\hat{y}_1 > \hat{y}_2 + margin$。
    * **新 Loss 公式**：
        $$\mathcal{L}_{total} = \mathcal{L}_{wEMD} + \lambda_1 \mathcal{L}_{atts} + \lambda_2 \mathcal{L}_{Rank}$$
        $$\mathcal{L}_{Rank} = \max(0, -\text{sign}(y_1 - y_2)(\hat{y}_1 - \hat{y}_2) + \text{margin})$$
    * **預期效果**：這將直接提升 **SRCC** 指標，這是美學評估中最具說服力的指標。

---

### 總結：建議的實驗路徑 (Experiment Roadmap)

如果你要指派工程師進行優化，建議依照以下順序，由簡入深：

1.  **Phase 1 (Baseline 升級)**:
    * 將 Backbone 換成 **Swin-Transformer Tiny/Small**。
    * 保留原有的 SAMP 模組邏輯。
    * **預期**：MSE 顯著下降，SRCC 小幅上升。

2.  **Phase 2 (Loss 調整)**:
    * 在 Phase 1 的基礎上，加入 **Ranking Loss**。
    * [cite_start]**預期**：SRCC 大幅上升 (可能突破論文中的 0.6564 [cite: 218])。

3.  **Phase 3 (架構重構 - 高風險高回報)**:
    * 移除 SAMP 的固定 8 模式，改用 **Self-Attention 機制** 或 **GNN** 來動態學習構圖佈局。
    * 將 Saliency Map 改為 Spatial Attention 注入。

這份計畫能夠讓你在論文的基礎上，有憑有據地進行改進。是否需要我針對「Ranking Loss」或「Swin Transformer 整合」提供更詳細的 PyTorch 程式碼片段？