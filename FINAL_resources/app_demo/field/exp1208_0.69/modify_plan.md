### 1\. 核心差異比較表 (Code vs. SAMP-Net)

| 特性 | SAMP-Net (原論文) | 你的實作 (Current Code) | 評價 |
| :--- | :--- | :--- | :--- |
| **Backbone** | ResNet18 (CNN) | **Swin-Tiny (Transformer)** | **你的勝出**。Backbone 更強，特徵提取能力更好。 |
| **Input Size** | 224 x 224 | **384 x 384** | **你的勝出**。構圖需要高解析度，這點做對了。 |
| **顯著性融合** | Flatten + Concat (向量拼接) | **Element-wise Multiplication** (特徵加權) | **平手/略差**。SAMP 雖然暴力拼接，但保留了顯著特徵的獨立性；你的乘法讓顯著特徵與原圖特徵「混在一起」，模型難以區分「這是背景」還是「這是顯著物體」。 |
| **構圖感知 (核心)** | **Multi-pattern Pooling** (8種網格切分) | **Global Average Pooling** (`mean(dim=[2, 3])`) | **嚴重缺失**。這是導致效能瓶頸的最大原因（詳見下文）。 |
| **Loss** | Weighted EMD + Attr | EMD + Attr + **Ranking** | **你的勝出**。Ranking Loss 理應帶來巨大提升，但被架構缺陷拖累。 |
| **Optimization** | 未提及細節 | **統一 LR (1e-4)** | **嚴重缺失**。對於 Pre-trained Transformer 來說，這個 LR 太大。 |

-----

### 2\. 為什麼你的模型會卡住？(批判性診斷)

#### 致命傷 1：Global Average Pooling 殺死了「構圖」

在你的 `model.py` 第 87 行：

```python
feat_global = features_attended.mean(dim=[2, 3])
```

這行程式碼執行了 **全局平均池化 (Global Average Pooling, GAP)**。

  * **問題所在**：構圖評估的本質是「**空間位置的安排**」。例如：「主體是否在三分線上？」、「畫面是否左右平衡？」。
  * **後果**：當你做 GAP 時，你把 $12 \times 12$ 的空間特徵圖直接壓縮成一個向量。**所有的空間資訊（左邊、右邊、上面、下面）在這一瞬間全部消失了。**
  * **SAMP-Net 的做法**：它**沒有**做 GAP。它設計了 8 種不同的 Mask（三分法網格、對稱網格等），分別去 Pool 不同的區域，告訴模型「請看看左上角和右下角的關係」。
  * **結論**：你的模型現在只知道「這張圖有什麼（Content）」，完全不知道「東西在哪裡（Layout）」。它變成了一個圖像分類器，而不是構圖評估器。

#### 致命傷 2：Learning Rate 策略錯誤 (Epoch 8 停滯的主因)

在 `config.py` 中，你設定 `LR = 1e-4`，並且在 `train.py` 中對所有參數（Backbone + Head）使用相同的 LR。

  * **問題所在**：Swin Transformer 是在 ImageNet 上預訓練過的。對於只有 9,000 張圖片的小資料集：
      * **Backbone ($1e-4$)**：這個學習率太大了。它會在前幾個 Epoch 就破壞掉 SwinT 預訓練好的特徵提取能力 (Catastrophic Forgetting)。
      * **Head ($1e-4$)**：這個學習率是合適的。
  * **現象解釋**：
      * **Epoch 1-8**：Head 快速學會了如何將特徵映射到分數，Backbone 稍微適應了新圖片，SRCC 快速衝到 0.67。
      * **Epoch 8+**：Backbone 的參數因為 LR 過大開始劇烈震盪或過擬合（Overfitting），破壞了原本良好的特徵，導致 Validation Loss 不降反升，SRCC 再也上不去。

#### 致命傷 3：死掉的程式碼 (Dead Code / Implementation Bug)

在 `model.py` 中，你定義了 `SpatialAttention` class，但是在 `SwinSAMPNet` 的 `forward` 函數中：

```python
# 你定義了 self.spatial_att = SpatialAttention() 嗎？並沒有。
# 且第 83 行直接做了簡單乘法：
features_attended = features * (1.0 + sal_down)
```

這意味著你原本可能想用 Conv+Sigmoid 來學習如何融合顯著性，但實際上你只用了最簡單的線性加權。這限制了模型學習「如何利用顯著性」的能力。

-----

### 3\. 優化方案 (Action Plan)

為了突破 0.67 並真正利用 SwinT 的優勢，請依照此規格修改程式碼：

#### Step 1: 修正模型架構 (取代 GAP)

不要把特徵圖平均掉。請引入一個能感知空間的模組。
**修改 `model.py`:**

```python
class SwinSAMPNet(nn.Module):
    def __init__(self, cfg):
        # ... (前段不變) ...
        self.feature_dim = 768
        
        # 1. 定義可學習的構圖 Queries (取代 SAMP 的固定網格)
        self.num_queries = 16  # 模擬 16 種潛在的構圖模式
        self.layout_queries = nn.Parameter(torch.randn(1, self.num_queries, self.feature_dim))
        
        # 2. Cross-Attention 模組
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=8, batch_first=True)
        
        # Heads 改為接受 Query 的結果
        self.score_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256), # 從 Queries 中聚合資訊
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Softmax(dim=1)
        )
        # ...

    def forward(self, x, saliency):
        # ... (Backbone 提取 features 得到 [B, 768, 12, 12]) ...
        
        # 融合 Saliency (保持你的做法或使用 SpatialAttention)
        features_attended = features * (1.0 + sal_down) 
        
        # [B, C, H, W] -> [B, H*W, C] (Sequence format for Attention)
        B, C, H, W = features_attended.shape
        feat_seq = features_attended.view(B, C, H*W).permute(0, 2, 1) # [B, 144, 768]
        
        # Expand Queries
        queries = self.layout_queries.expand(B, -1, -1) # [B, 16, 768]
        
        # Cross Attention: 
        # Query找尋圖片中符合特定構圖模式的區域
        attn_out, _ = self.cross_attn(query=queries, key=feat_seq, value=feat_seq)
        
        # attn_out: [B, 16, 768]
        # 現在我們有 16 個代表不同構圖特徵的向量，將它們平均或透過 Attention Pooling 融合
        feat_layout = attn_out.mean(dim=1) # [B, 768] -> 這裡的 Mean 是聚合模式，不是聚合空間，是 OK 的
        
        score_dist = self.score_head(feat_layout)
        # ...
```

**原理**：這讓模型通過 Attention 機制去「掃描」整張圖的 $12 \times 12$ 區域，保留了空間互動的資訊。

#### Step 2: 修正訓練策略 (分層 LR)

**修改 `train.py` 中的 Optimizer 設定:**

```python
# 將參數分組
backbone_params = list(map(id, model.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': cfg.LR * 0.1}, # Backbone 用 1e-5
    {'params': base_params, 'lr': cfg.LR}                        # Head 用 1e-4
], weight_decay=cfg.WEIGHT_DECAY)
```

這能讓 Backbone 微調而不被破壞，同時讓新的 Head 快速學習。

#### Step 3: 修正資料增強 (Data Augmentation)

在 `dataset.py` 中，你只用了 Resize + Normalize。
對於 9,000 張圖片，這一定會 Overfit。請加入 **水平翻轉 (Horizontal Flip)**。

```python
self.transform = transforms.Compose([
    transforms.Resize((self.image_size, self.image_size)),
    transforms.RandomHorizontalFlip(p=0.5), # 構圖評估中，左右翻轉通常不改變好壞（除非有文字）
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### 總結

目前的架構只發揮了 Swin Transformer 的「特徵提取」能力，卻丟棄了構圖評估最需要的「空間結構」資訊。

請優先實作 **Step 1 (Cross-Attention 取代 GAP)** 和 **Step 2 (分層 LR)**，你應該能看到 SRCC 突破 0.67 並穩定上升。