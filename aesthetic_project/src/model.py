import torch
import torch.nn as nn
import timm

class AestheticViT(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        super(AestheticViT, self).__init__()
        
        # 載入預訓練的 ViT
        # num_classes=0 returns the pooled features (representation)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.2
        )
        
        # 取得特徵維度 (Tiny ViT 通常是 192, Small 是 384, Base 是 768)
        embed_dim = self.backbone.num_features
        
        # 建立 6 個獨立的 Regression Head
        # Output 0: IAS (Total Score)
        # Output 1-4: Sub-scores (C, L, F, O)
        self.head_ias = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid())
        self.head_c = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_l = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_f = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_o = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        
        # 預測各項分數
        ias = self.head_ias(features)
        c = self.head_c(features)
        l = self.head_l(features)
        f = self.head_f(features)
        o = self.head_o(features)
        
        # 拼接輸出: [Batch, 5]
        return torch.cat([ias, c, l, f, o], dim=1)

# 你可以保留原本的 AestheticViT，新增這個類別以便進行比較
class AestheticSwin(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super(AestheticSwin, self).__init__()
        
        print(f"🏗️ Building Swin Transformer: {model_name}")
        
        # 載入 Swin Transformer
        # 注意：Swin 同樣支援 drop_path_rate 等正則化參數
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,        # 移除分類頭，只取特徵
            drop_rate=0.1,        # Head Dropout
            attn_drop_rate=0.1,   # Attention Dropout
            drop_path_rate=0.2    # Stochastic Depth (防過擬合關鍵)
        )
        
        # 自動取得特徵維度 (Swin Tiny 通常是 768)
        embed_dim = self.backbone.num_features
        print(f"ℹ️ Feature Dimension: {embed_dim}")
        
        # 建立 5 個獨立的 Regression Heads (與 ViT 版本保持一致，公平比較)
        self.head_ias = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid())
        self.head_c = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_l = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_f = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_o = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🥶 Backbone frozen.")
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen.")

    def forward(self, x):
        features = self.backbone(x)
        
        # 預測各項分數
        ias = self.head_ias(features)
        c = self.head_c(features)
        l = self.head_l(features)
        f = self.head_f(features)
        o = self.head_o(features)
        
        return torch.cat([ias, c, l, f, o], dim=1)


