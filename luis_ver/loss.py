import torch
import torch.nn as nn

class RankLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, preds, targets):
        # preds: [B, 1]
        # targets: [B]
        
        n = preds.size(0)
        if n < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        # Expand to matrices [B, B]
        preds_i = preds.expand(n, n)
        preds_j = preds.t().expand(n, n)
        
        targets_i = targets.unsqueeze(1).expand(n, n)
        targets_j = targets.unsqueeze(0).expand(n, n)
        
        # Differences
        diff_preds = preds_i - preds_j
        diff_targets = targets_i - targets_j
        
        # Sign of target difference
        target_sign = torch.sign(diff_targets)
        
        # Loss
        # max(0, margin - sign * (pred_i - pred_j))
        loss = torch.relu(self.margin - target_sign * diff_preds)
        
        # Average over all pairs
        return loss.mean()

class TotalLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=5.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rank_loss = RankLoss(margin=0.1) # Small margin for stability
        self.reg_loss = nn.HuberLoss()
        
    def forward(self, outputs, targets):
        # outputs: dict of preds [B, 1]
        # targets: dict of targets [B]
        
        # Rank Loss for IAS
        loss_rank = self.rank_loss(outputs['IAS'], targets['IAS'])
        
        # Regression Loss for C, L, F, O
        loss_c = self.reg_loss(outputs['C'].squeeze(), targets['C'])
        loss_l = self.reg_loss(outputs['L'].squeeze(), targets['L'])
        loss_f = self.reg_loss(outputs['F'].squeeze(), targets['F'])
        loss_o = self.reg_loss(outputs['O'].squeeze(), targets['O'])
        
        loss_reg = (loss_c + loss_l + loss_f + loss_o) / 4.0 # Average reg loss
        
        total_loss = self.lambda1 * loss_rank + self.lambda2 * loss_reg
        
        return total_loss, {'rank': loss_rank, 'reg': loss_reg}
