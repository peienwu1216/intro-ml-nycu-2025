import torch
import torch.nn as nn
import torch.nn.functional as F

class EMDLoss(nn.Module):
    def __init__(self, r=2, reduction='mean'):
        super(EMDLoss, self).__init__()
        self.r = r
        self.reduction = reduction

    def forward(self, target, prediction, weight=None):
        # target: [B, 5] (distribution)
        # prediction: [B, 5] (distribution)
        # weight: [B] (optional sample weight)
        
        cdf_target = torch.cumsum(target, dim=1)
        cdf_pred = torch.cumsum(prediction, dim=1)
        
        diff = torch.abs(cdf_target - cdf_pred)
        loss = torch.pow(diff, self.r).mean(dim=1)
        loss = torch.pow(loss, 1./self.r)
        
        if weight is not None:
            loss = loss * weight
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class RankLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(RankLoss, self).__init__()
        self.margin = margin
        
    def forward(self, preds, targets):
        # preds: [B] (mean scores)
        # targets: [B] (mean scores)
        
        n = preds.size(0)
        if n < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        # Expand to matrices [B, B]
        preds_i = preds.unsqueeze(1).expand(n, n)
        preds_j = preds.unsqueeze(0).expand(n, n)
        
        targets_i = targets.unsqueeze(1).expand(n, n)
        targets_j = targets.unsqueeze(0).expand(n, n)
        
        # Differences
        diff_preds = preds_i - preds_j
        diff_targets = targets_i - targets_j
        
        # Sign of target difference
        # We only care about pairs where target difference is non-zero
        target_sign = torch.sign(diff_targets)
        mask = target_sign != 0
        
        # Loss: max(0, -sign * (pred_i - pred_j) + margin)
        # If target_i > target_j (sign=1), we want pred_i > pred_j + margin
        # => -(pred_i - pred_j) + margin < 0
        # => pred_i - pred_j > margin
        
        loss = torch.relu(-target_sign * diff_preds + self.margin)
        
        # Apply mask and average
        loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)
        
        return loss

class AttributeLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(AttributeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.weight = weight
        
    def forward(self, preds, targets):
        return self.weight * self.mse(preds, targets)
