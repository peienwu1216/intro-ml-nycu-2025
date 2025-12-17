import torch
import torch.nn as nn
import torch.nn.functional as F

class EMDLoss(nn.Module):
    """
    [Control Variable] EMD Loss
    Standard Earth Mover's Distance loss used in SAMPNet.
    """
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
    """
    [Control Variable] Rank Loss
    Pairwise ranking loss to optimize SRCC directly.
    """
    def __init__(self, margin=0.05):
        super(RankLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
        
    def forward(self, preds, targets):
        # preds: [B] (mean scores)
        # targets: [B] (mean scores)
        
        n = preds.size(0)
        if n < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        # Create pairs
        preds_i = preds.unsqueeze(1).expand(n, n).reshape(-1)
        preds_j = preds.unsqueeze(0).expand(n, n).reshape(-1)
        
        targets_i = targets.unsqueeze(1).expand(n, n).reshape(-1)
        targets_j = targets.unsqueeze(0).expand(n, n).reshape(-1)
        
        diff_targets = targets_i - targets_j
        target_labels = torch.sign(diff_targets)
        
        # Filter out pairs where targets are equal (sign is 0)
        mask = target_labels != 0
        
        if mask.sum() == 0:
             return torch.tensor(0.0, device=preds.device, requires_grad=True)
             
        preds_i_masked = preds_i[mask]
        preds_j_masked = preds_j[mask]
        target_labels_masked = target_labels[mask]
        
        return self.loss_fn(preds_i_masked, preds_j_masked, target_labels_masked)

class AttributeLoss(nn.Module):
    """
    [Control Variable] Attribute Loss
    MSE loss for auxiliary attributes.
    """
    def __init__(self, weight=1.0):
        super(AttributeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.weight = weight
        
    def forward(self, preds, targets):
        return self.weight * self.mse(preds, targets)
