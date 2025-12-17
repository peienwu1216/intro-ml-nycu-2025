import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GRN(nn.Module):
    """ Global Response Normalization layer
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, H, W, C]
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x

class SGFM(nn.Module):
    """ Saliency-Guided Feature Modulation """
    def __init__(self, dim):
        super().__init__()
        self.conv_gamma = nn.Conv2d(1, dim, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(1, dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, saliency):
        # x: [B, C, H, W]
        # saliency: [B, 1, H, W]
        
        # Resize saliency to match x
        saliency = F.interpolate(saliency, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        gamma = self.sigmoid(self.conv_gamma(saliency))
        beta = self.conv_beta(saliency)
        
        return x * (1 + gamma) + beta

class GRNAwareAttentionPooling(nn.Module):
    """ GRN-Aware Attention Pooling """
    def __init__(self, dim):
        super().__init__()
        self.spatial_mixing = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.grn = GRN(dim)
        self.attn_conv = nn.Conv2d(dim, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. Spatial Mixing
        out = self.spatial_mixing(x)
        
        # 2. Feature Competition (GRN)
        # GRN expects [B, H, W, C]
        out = out.permute(0, 2, 3, 1)
        out = self.grn(out)
        out = out.permute(0, 3, 1, 2) # Back to [B, C, H, W]
        
        # 3. Attention Map
        attn_map = self.attn_conv(out) # [B, 1, H, W]
        
        # Spatial Softmax
        B, _, H, W = attn_map.shape
        attn_map = attn_map.view(B, 1, -1)
        attn_map = F.softmax(attn_map, dim=-1)
        attn_map = attn_map.view(B, 1, H, W)
        
        # 4. Weighted Pooling
        x_weighted = x * attn_map
        return x_weighted.sum(dim=(2, 3)) # [B, C]

class SwinSAMPNet(nn.Module):
    def __init__(self, cfg):
        super(SwinSAMPNet, self).__init__()
        
        # Backbone: ConvNeXt V2 Nano
        self.backbone = timm.create_model('convnextv2_nano', pretrained=True, features_only=True)
        
        # Feature Dimensions (Nano)
        # Stage 3: 320, Stage 4: 640
        self.dim_s3 = 320
        self.dim_s4 = 640
        self.target_dim = 512
        
        # Adaptation Layers
        self.adapt_s3 = nn.Sequential(
            nn.Conv2d(self.dim_s3, self.target_dim, 1),
            nn.GroupNorm(1, self.target_dim) # LayerNorm equivalent for 2D
        )
        self.adapt_s4 = nn.Sequential(
            nn.Conv2d(self.dim_s4, self.target_dim, 1),
            nn.GroupNorm(1, self.target_dim)
        )
        
        # Modules
        self.sgfm = SGFM(self.target_dim)
        self.pooling = GRNAwareAttentionPooling(self.target_dim)
        
        # Heads
        self.score_head = nn.Sequential(
            nn.Linear(self.target_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, 5), # 5-class distribution
            nn.Softmax(dim=1)
        )
        
        self.attr_head = nn.Sequential(
            nn.Linear(self.target_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, cfg.NUM_ATTRIBUTES),
            nn.Sigmoid()
        )
        
        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, saliency):
        # x: [B, 3, 384, 384]
        # saliency: [B, 1, 384, 384]
        
        # 1. Backbone Features
        features = self.backbone(x)
        feat_s3 = features[2] # Stage 3
        feat_s4 = features[3] # Stage 4
        
        # 2. Adaptation & Fusion
        feat_s3 = self.adapt_s3(feat_s3)
        feat_s4 = self.adapt_s4(feat_s4)
        
        # Upsample S4 to S3 size
        feat_s4_up = F.interpolate(feat_s4, size=feat_s3.shape[2:], mode='bilinear', align_corners=False)
        feat_fused = feat_s3 + feat_s4_up
        
        # Register hook for Grad-CAM
        if feat_fused.requires_grad:
            h = feat_fused.register_hook(self.activations_hook)
            self.activations = feat_fused
        
        # 3. SGFM
        feat_modulated = self.sgfm(feat_fused, saliency)
        
        # 4. GRN-Aware Attention Pooling
        feat_global = self.pooling(feat_modulated)
        
        # 5. Heads
        score_dist = self.score_head(feat_global)
        attrs = self.attr_head(feat_global)
        
        return score_dist, attrs

    # Grad-CAM method
    def get_grad_cam(self, target_layer_output, target_class_index=None):
        # This is a simplified Grad-CAM implementation
        # Call this after backward()
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None
            
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam
