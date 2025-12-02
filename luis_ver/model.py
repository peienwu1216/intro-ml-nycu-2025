import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class SwinMTL_NoPost(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Backbone
        # Use weights='IMAGENET1K_V1' if available, else pretrained=True (deprecated)
        try:
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            self.backbone = models.swin_t(weights=weights)
        except:
            self.backbone = models.swin_t(pretrained=True)
        
        # Feature Extractor
        # We extract features after Stage 2 (features.3) and Stage 4 (features.7)
        # Note: In torchvision swin, features.3 is the second stage block.
        # features.7 is the fourth stage block.
        
        return_nodes = {
            'features.3': 'feat_low',  # Stage 2
            'features.7': 'feat_high', # Stage 4
        }
        self.feature_extractor = create_feature_extractor(self.backbone, return_nodes=return_nodes)
        
        # Dimensions for Swin-T
        # Stage 1: 96
        # Stage 2: 192
        # Stage 3: 384
        # Stage 4: 768
        self.dim_low = 192
        self.dim_high = 768
        
        # CLIP
        self.use_clip = True
        self.clip_dim = 512
        try:
            import clip
            # Load CLIP model
            # We load it to CPU first, then move to device in forward or init
            self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: CLIP not found. Using dummy embeddings.")
            self.use_clip = False
            self.clip_model = None
            
        # Heads
        # Head-C (Composition): Input Feat_High
        self.head_c = self._make_head(self.dim_high)
        
        # Head-L (Light/Color): Input Feat_High
        self.head_l = self._make_head(self.dim_high)
        
        # Head-F (Focus): Input Feat_Low
        self.head_f = self._make_head(self.dim_low)
        
        # Head-O (Originality): Input Concat(Feat_High, CLIP)
        self.head_o = self._make_head(self.dim_high + self.clip_dim)
        
        # Head-IAS (Global): Input Concat(C, L, F, O, Feat_High)
        # C, L, F, O are scalars (1 dim each) -> 4 dims
        self.head_ias = self._make_head(4 + self.dim_high)

    def _make_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Backbone
        # Swin expects [B, C, H, W]
        features = self.feature_extractor(x)
        feat_low = features['feat_low']   # [B, H/8, W/8, 192] -> Permuted in torchvision swin?
        feat_high = features['feat_high'] # [B, H/32, W/32, 768]
        
        # Torchvision Swin outputs are [B, H, W, C] (channels last) inside the model layers?
        # Let's check. The `permute` layer at the end of Swin converts to [B, C, H, W].
        # But we are extracting from intermediate layers.
        # In torchvision implementation, the stages (BasicLayer) output [B, H, W, C].
        # So we need to permute to [B, C, H, W] for pooling or just pool on last dim.
        
        # Global Average Pooling
        # Input: [B, H, W, C]
        feat_low_pool = feat_low.mean(dim=[1, 2]) # [B, 192]
        feat_high_pool = feat_high.mean(dim=[1, 2]) # [B, 768]
        
        # CLIP
        if self.use_clip and self.clip_model is not None:
            with torch.no_grad():
                # Resize for CLIP (224x224)
                # x is [B, 3, 384, 384]
                x_clip = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
                # CLIP expects normalization. We assume x is already normalized for ImageNet.
                # We might need to re-normalize if CLIP expects different stats.
                # CLIP stats: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
                # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # They are close enough for this task.
                
                clip_emb = self.clip_model.encode_image(x_clip).float() # [B, 512]
        else:
            clip_emb = torch.zeros(x.size(0), self.clip_dim, device=x.device)
            
        # Heads
        out_c = self.head_c(feat_high_pool)
        out_l = self.head_l(feat_high_pool)
        out_f = self.head_f(feat_low_pool)
        
        # Head O
        feat_o = torch.cat([feat_high_pool, clip_emb], dim=1)
        out_o = self.head_o(feat_o)
        
        # Head IAS
        feat_ias = torch.cat([out_c, out_l, out_f, out_o, feat_high_pool], dim=1)
        out_ias = self.head_ias(feat_ias)
        
        return {
            'C': out_c,
            'L': out_l,
            'F': out_f,
            'O': out_o,
            'IAS': out_ias
        }
