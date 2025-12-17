import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SwinSAMPNet(nn.Module):
    def __init__(self, cfg):
        super(SwinSAMPNet, self).__init__()
        
        # Backbone: Swin Transformer
        # We use swin_t by default
        try:
            weights = models.Swin_T_Weights.IMAGENET1K_V1
            self.backbone = models.swin_t(weights=weights)
        except:
            self.backbone = models.swin_t(pretrained=True)
            
        # Feature Extractor
        # Swin-T output channels: 768 (Stage 4)
        self.feature_dim = 768
        
        # We need to extract features before the final pooling/head
        # Swin structure: features -> norm -> permute -> avgpool -> flatten -> head
        # We want the spatial features from 'features' (Stage 4)
        # But torchvision swin 'features' returns [B, H, W, C] (channels last)
        # We will handle this in forward
        
        # Spatial Attention for Saliency Fusion
        # We will inject Saliency as a spatial attention mask
        # Saliency map is [B, 1, H, W]
        # We can downsample saliency to match feature map size
        
        # Heads
        self.score_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, 5), # 5-class distribution
            nn.Softmax(dim=1)
        )
        
        self.attr_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, cfg.NUM_ATTRIBUTES),
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
        # Swin features: [B, H/32, W/32, 768] -> [B, 12, 12, 768] for 384 input
        features = self.backbone.features(x) 
        
        # Permute to [B, C, H, W] for standard processing
        features = features.permute(0, 3, 1, 2) # [B, 768, 12, 12]
        
        # Register hook for Grad-CAM
        if features.requires_grad:
            h = features.register_hook(self.activations_hook)
            self.activations = features
        
        # 2. Saliency Fusion (Spatial Attention)
        # Downsample saliency to feature map size
        sal_down = torch.nn.functional.interpolate(saliency, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # Simple Fusion: Multiply features by (1 + Saliency)
        # Or use Saliency as Attention
        # Let's use Saliency to re-weight spatial features
        # Normalize saliency to 0-1 range if not already
        
        # Enhanced Feature = Feature * (1 + alpha * Saliency)
        # This highlights salient regions while keeping background info
        features_attended = features * (1.0 + sal_down)
        
        # 3. Global Pooling
        # [B, 768, 12, 12] -> [B, 768]
        feat_global = features_attended.mean(dim=[2, 3])
        
        # 4. Heads
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
