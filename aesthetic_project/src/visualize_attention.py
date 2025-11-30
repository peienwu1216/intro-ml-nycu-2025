import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import AestheticViT
from config import Config

class RegressionOutputTarget:
    def __init__(self, index):
        self.index = index
    
    def __call__(self, model_output):
        if model_output.dim() == 2:
            return model_output[:, self.index]
        elif model_output.dim() == 1:
            return model_output[self.index]
        return model_output

def reshape_transform(tensor, height=14, width=14):
    # ViT output shape: [Batch, Patches+1, Embed_Dim]
    # Need to discard CLS token (index 0) and reshape to [Batch, Embed_Dim, Height, Width]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_attention(image_path, model, device, save_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # 1. Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Prepare image for visualization (float32, 0-1 range)
    rgb_img = np.float32(img) / 255
    
    # 2. Initialize Grad-CAM
    # Target layer: The last normalization layer of the last block in ViT
    target_layers = [model.backbone.blocks[-1].norm1]
    
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    # 3. Generate CAM
    # We want to explain the "IAS" score (index 0)
    targets = [RegressionOutputTarget(0)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 4. Overlay on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 5. Save result
    # Convert RGB to BGR for cv2 saving
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, visualization)
    print(f"Saved attention map to {save_path}")

def main():
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load Model
    model = AestheticViT(model_name=Config.MODEL_NAME).to(device)
    checkpoint_path = "aesthetic_vit_model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Checkpoint not found! Please train the model first.")
        return
        
    model.eval()
    
    # Paths
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    attention_maps_dir = os.path.join(figures_dir, 'attention_maps')
    os.makedirs(attention_maps_dir, exist_ok=True)
    
    # Process Validation Sets (set1 to set10)
    validation_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../validation"))
    
    if not os.path.exists(validation_dir):
        print(f"Validation directory not found: {validation_dir}")
        return
        
    for set_name in sorted(os.listdir(validation_dir)):
        set_path = os.path.join(validation_dir, set_name)
        if os.path.isdir(set_path) and set_name.startswith("set"):
            print(f"Processing {set_name}...")
            
            # Create output directory for this set
            output_set_dir = os.path.join(attention_maps_dir, set_name)
            os.makedirs(output_set_dir, exist_ok=True)
            
            # Process all images in the set folder
            for file_name in os.listdir(set_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(set_path, file_name)
                    
                    # Construct output filename (keep original name but change ext to png)
                    base_name = os.path.splitext(file_name)[0]
                    output_filename = f"{base_name}.png"
                    output_path = os.path.join(output_set_dir, output_filename)
                    
                    print(f"  Processing {file_name} -> {output_filename}")
                    visualize_attention(img_path, model, device, output_path)


if __name__ == "__main__":
    main()

