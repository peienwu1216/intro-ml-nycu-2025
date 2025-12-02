import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from model import SwinMTL_NoPost
from dataset import LetterboxPad

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = SwinMTL_NoPost()
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model, image_paths, device):
    # Transform matches training
    transform = transforms.Compose([
        LetterboxPad(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Collect all files first to avoid recursive printing of headers
    files_to_process = []
    for path in image_paths:
        if os.path.isdir(path):
            subfiles = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            subfiles.sort()
            files_to_process.extend(subfiles)
        elif os.path.isfile(path):
             files_to_process.append(path)
    
    if not files_to_process:
        print("No images found.")
        return

    print("\n" + "="*90)
    print(f"{'Image Name':<30} | {'IAS':<8} | {'Comp':<8} | {'Light':<8} | {'Focus':<8} | {'Orig':<8}")
    print("-" * 90)

    for img_path in files_to_process:
        if not os.path.exists(img_path):
            print(f"{'File not found':<30} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            continue
            
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                
            ias = outputs['IAS'].item()
            c = outputs['C'].item()
            l = outputs['L'].item()
            f = outputs['F'].item()
            o = outputs['O'].item()
            
            filename = os.path.basename(img_path)
            # Truncate filename if too long
            display_name = filename
            if len(display_name) > 28:
                display_name = display_name[:25] + "..."
                
            print(f"{display_name:<30} | {ias:.4f}   | {c:.4f}   | {l:.4f}   | {f:.4f}   | {o:.4f}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
    print("="*90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Aesthetics using Swin-MTL model")
    parser.add_argument('images', nargs='+', help='Path to input image(s) or directory')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model weights (default: best_model.pth)')
    parser.add_argument('--device', type=str, default='', help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Check model path
    model_path = args.model
    if not os.path.exists(model_path):
        # Try looking in the current directory if not found
        current_dir_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
        if os.path.exists(current_dir_model):
            model_path = current_dir_model
        else:
            print(f"Error: Model file not found at {args.model}")
            print("Please train the model first using 'python3 train.py' or specify the correct path.")
            return

    try:
        model = load_model(model_path, device)
        predict(model, args.images, device)
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == '__main__':
    main()
