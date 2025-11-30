import torch
import torch.nn as nn
import timm

class AestheticViT(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        super(AestheticViT, self).__init__()
        
        # 載入預訓練的 ViT
        # num_classes=0 returns the pooled features (representation)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 取得特徵維度 (Tiny ViT 通常是 192, Small 是 384, Base 是 768)
        embed_dim = self.backbone.num_features
        
        # 建立 6 個獨立的 Regression Head
        # Output 0: IAS (Total Score)
        # Output 1-5: Sub-scores (C, L, F, P, O)
        self.head_ias = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid())
        self.head_c = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_l = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_f = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_p = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        self.head_o = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid())
        
    def forward(self, x):
        features = self.backbone(x)
        
        # 預測各項分數
        ias = self.head_ias(features)
        c = self.head_c(features)
        l = self.head_l(features)
        f = self.head_f(features)
        p = self.head_p(features)
        o = self.head_o(features)
        
        # 拼接輸出: [Batch, 6]
        return torch.cat([ias, c, l, f, p, o], dim=1)

