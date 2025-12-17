import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import sys


class GradCAM:
    """
    GradCAM implementation for visualizing model attention.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, saliency_tensor, target_score=None):
        """
        Generate GradCAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            saliency_tensor: Saliency map tensor [1, 1, H, W]
            target_score: If None, uses the predicted aesthetic score
        
        Returns:
            cam: GradCAM heatmap [H, W] normalized to [0, 1]
        """
        self.model.zero_grad()
        
        # Forward pass
        pred_dist, pred_attrs = self.model(input_tensor, saliency_tensor)
        
        # Use the weighted mean score as target (aesthetic score)
        # Score = sum(i * p_i) for i in 1..5
        weights = torch.arange(1, 6, dtype=torch.float32, device=pred_dist.device)
        score = (pred_dist * weights).sum()
        
        # Backward pass
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def apply_colormap(cam, original_image, alpha=0.5):
    """
    Apply colormap to GradCAM and overlay on original image.
    
    Args:
        cam: GradCAM heatmap [H, W] normalized to [0, 1]
        original_image: Original PIL Image
        alpha: Overlay transparency
    
    Returns:
        overlay: Combined image as numpy array (RGB)
    """
    # Resize CAM to match original image size
    orig_size = original_image.size  # (W, H)
    cam_resized = cv2.resize(cam, orig_size)
    
    # Convert to heatmap (JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy
    orig_np = np.array(original_image)
    
    # Overlay
    overlay = (1 - alpha) * orig_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, heatmap

# Add parent directory to path to import modules from field/
# This allows running the script from field/test_format/ or field/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from config import Config
    from dataset import detect_saliency, IMAGE_NET_MEAN, IMAGE_NET_STD
    from utils import dist2ave
    # Try to import from local model.py if models.convnext_v2 fails
    try:
        from models.convnext_v2 import ConvNeXtV2SAMPNet
    except ImportError:
        from model import SwinSAMPNet as ConvNeXtV2SAMPNet
except ImportError as e:
    # Fallback if running from root or other structure
    print(f"Warning: Import failed ({e}). Trying to adjust path...")
    # If running from root, field might be a package
    try:
        from field.config import Config
        from field.dataset import detect_saliency, IMAGE_NET_MEAN, IMAGE_NET_STD
        from field.utils import dist2ave
        from field.model import SwinSAMPNet as ConvNeXtV2SAMPNet
    except ImportError as e2:
        print(f"Error importing modules: {e2}")
        sys.exit(1)

def load_model(model_path, device, cfg):
    print(f"Loading model from {model_path}...")
    # Initialize model structure
    # Note: If you change the model in train.py, you must update this line too
    model = ConvNeXtV2SAMPNet(cfg)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model):
    """
    Get the target layer for GradCAM.
    For ConvNeXt V2, we use the last stage of the backbone.
    """
    # Try different model structures
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        # ConvNeXt V2 structure: backbone.stages[-1] or backbone.norm
        if hasattr(backbone, 'stages'):
            return backbone.stages[-1]
        elif hasattr(backbone, 'features'):
            return backbone.features[-1]
        elif hasattr(backbone, 'layer4'):
            return backbone.layer4
        else:
            # Fallback: try to get the last child
            children = list(backbone.children())
            if children:
                return children[-1]
    
    # For Swin Transformer
    if hasattr(model, 'swin'):
        return model.swin.layers[-1]
    
    # Fallback
    raise ValueError("Could not find target layer for GradCAM")


def predict(model, image_paths, device, cfg, save_gradcam=True, output_dir=None):
    # Transform matches training (Validation transform)
    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
    ])

    # Collect all files recursively
    files_to_process = []
    for path in image_paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        files_to_process.append(os.path.join(root, f))
        elif os.path.isfile(path):
             files_to_process.append(path)
    
    if not files_to_process:
        print("No images found.")
        return

    files_to_process.sort()
    
    # Setup GradCAM
    gradcam = None
    if save_gradcam:
        try:
            target_layer = get_target_layer(model)
            gradcam = GradCAM(model, target_layer)
            print(f"GradCAM enabled. Target layer: {target_layer.__class__.__name__}")
        except Exception as e:
            print(f"Warning: Could not initialize GradCAM: {e}")
            save_gradcam = False

    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)

    for img_path in files_to_process:
        if not os.path.exists(img_path):
            continue
            
        try:
            # Load Image
            image = Image.open(img_path).convert('RGB')
            
            # Saliency Detection (Same as training)
            src_np = np.asarray(image).copy()
            sal_map = detect_saliency(src_np, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            sal_tensor = torch.from_numpy(sal_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
            
            # Image Transform
            img_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, H, W]
            
            # Generate GradCAM if enabled (requires grad)
            cam = None
            if save_gradcam and gradcam is not None:
                # Need to enable grad for GradCAM
                img_tensor.requires_grad_(True)
                try:
                    cam = gradcam.generate(img_tensor, sal_tensor)
                except Exception as e:
                    print(f"  Warning: GradCAM failed: {e}")
                    cam = None
                finally:
                    img_tensor.requires_grad_(False)
            
            with torch.no_grad():
                pred_dist, pred_attrs = model(img_tensor, sal_tensor)
                pred_score = dist2ave(pred_dist).item()
            
            # Save GradCAM visualization
            if cam is not None:
                # Determine output path
                img_dir = os.path.dirname(img_path)
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                
                if output_dir:
                    gradcam_dir = output_dir
                else:
                    gradcam_dir = os.path.join(img_dir, "gradcam")
                
                os.makedirs(gradcam_dir, exist_ok=True)
                
                # Generate overlay
                overlay, heatmap = apply_colormap(cam, image, alpha=0.5)
                
                # Save overlay image
                overlay_path = os.path.join(gradcam_dir, f"{img_name}_gradcam.jpg")
                Image.fromarray(overlay).save(overlay_path, quality=95)
                
                # Also save pure heatmap
                heatmap_path = os.path.join(gradcam_dir, f"{img_name}_heatmap.jpg")
                Image.fromarray(heatmap).save(heatmap_path, quality=95)
                
            # Process Distribution
            dist_probs = pred_dist.squeeze().cpu().numpy()
            dist_str = "[" + ", ".join([f"{p:.3f}" for p in dist_probs]) + "]"

            # Process Attributes
            attrs = pred_attrs.squeeze().cpu().numpy()
            attr_names = cfg.ATTRIBUTE_TYPES
            
            # Display relative path if possible for cleaner output
            display_name = img_path
            try:
                # Try to make it relative to the first argument if it's a directory
                # Or just relative to CWD
                display_name = os.path.relpath(img_path)
            except:
                pass
                
            print(f"\nImage: {display_name}")
            print(f"  Aesthetic Score: {pred_score:.4f} (1-5)")
            print(f"  Score Distribution: {dist_str}")
            if cam is not None:
                print(f"  GradCAM saved to: {gradcam_dir}/")
            print(f"  Attributes:")
            for name, val in zip(attr_names, attrs):
                print(f"    - {name:<20}: {val:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Inference for Image Aesthetic Assessment")
    parser.add_argument('images', nargs='+', help='Path to input image(s) or directory')
    parser.add_argument('--model', type=str, default='checkpoints/ConvNeXtV2_SGFM_GRN/best_model.pth', help='Path to model weights')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--no-gradcam', action='store_true', help='Disable GradCAM generation')
    parser.add_argument('--output-dir', type=str, default='', help='Output directory for GradCAM images (default: gradcam/ in image folder)')
    
    args = parser.parse_args()
    
    cfg = Config()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Check model path
    model_path = args.model
    if not os.path.exists(model_path):
        # Try relative to script location
        current_dir_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
        if os.path.exists(current_dir_model):
            model_path = current_dir_model
        else:
            # Try relative to field/ directory
            field_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            potential_path = os.path.join(field_dir, args.model)
            if os.path.exists(potential_path):
                model_path = potential_path
            else:
                print(f"Error: Model file not found at {args.model}")
                return

    try:
        model = load_model(model_path, device, cfg)
        predict(model, args.images, device, cfg, 
                save_gradcam=not args.no_gradcam,
                output_dir=args.output_dir if args.output_dir else None)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
