import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    [Interface] Base Model
    All new models should inherit from this or follow this structure
    to ensure compatibility with the training loop.
    """
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        
    def forward(self, x, saliency):
        """
        Args:
            x: [B, 3, H, W] Input image
            saliency: [B, 1, H, W] Saliency map
            
        Returns:
            score_dist: [B, 5] Predicted score distribution (Softmax)
            attrs: [B, Num_Attrs] Predicted attributes (Sigmoid)
        """
        raise NotImplementedError
        
    def get_grad_cam(self, target_layer_output, target_class_index=None):
        """
        Optional: For visualization
        """
        return None
