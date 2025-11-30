import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from model import AestheticViT
from config import Config

# 設定中文字型 (如果有的話，或是使用英文標籤)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
# plt.rcParams['axes.unicode_minus'] = False

def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Output order: [IAS, C, L, F, P, O]
        scores = outputs.cpu().numpy()[0]
        
    return scores

def scan_validation_sets(validation_dir):
    """Scans the validation directory for set folders and image pairs."""
    sets_data = []
    
    if not os.path.exists(validation_dir):
        print(f"Validation directory not found: {validation_dir}")
        return []
        
    set_folders = sorted([d for d in os.listdir(validation_dir) if d.startswith("set") and os.path.isdir(os.path.join(validation_dir, d))])
    
    # Sort numerically if possible (set1, set2, ..., set10)
    try:
        set_folders.sort(key=lambda x: int(x.replace("set", "")))
    except:
        pass # Keep lexicographical sort if fails

    for folder in set_folders:
        folder_path = os.path.join(validation_dir, folder)
        files = os.listdir(folder_path)
        
        good_img = next((f for f in files if "good" in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
        bad_img = next((f for f in files if "bad" in f.lower() and f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
        
        if good_img and bad_img:
            sets_data.append({
                "set_name": folder,
                "good_path": os.path.join(folder_path, good_img),
                "bad_path": os.path.join(folder_path, bad_img)
            })
    
    return sets_data

def visualize_results(results):
    """
    results: list of dicts containing set_name and scores for good/bad images
    """
    if not results:
        return
        
    # Ensure figures directory exists
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    set_names = [r['set_name'] for r in results]
    
    # Extract IAS scores
    good_ias = [r['good_scores'][0] for r in results]
    bad_ias = [r['bad_scores'][0] for r in results]
    
    # 1. Bar Chart: Good vs Bad IAS
    x = np.arange(len(set_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, good_ias, width, label='Good Image', color='skyblue')
    plt.bar(x + width/2, bad_ias, width, label='Bad Image', color='salmon')
    
    plt.xlabel('Image Sets')
    plt.ylabel('Predicted IAS Score (0-1)')
    plt.title('Aesthetic Assessment: Good vs Bad Images across Sets')
    plt.xticks(x, set_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path_ias = os.path.join(figures_dir, "inference_ias_comparison.png")
    plt.savefig(save_path_ias)
    print(f"Saved IAS comparison chart to {save_path_ias}")
    
    # 2. Radar Chart for Average Attributes
    # Attributes: C, L, F, P, O (indices 1-5)
    labels = ['Composition (C)', 'Lighting (L)', 'Focus (F)', 'Post-proc (P)', 'Originality (O)']
    
    avg_good_attrs = np.mean([r['good_scores'][1:] for r in results], axis=0)
    avg_bad_attrs = np.mean([r['bad_scores'][1:] for r in results], axis=0)
    
    # Radar chart setup
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    avg_good_attrs = np.concatenate((avg_good_attrs, [avg_good_attrs[0]]))
    avg_bad_attrs = np.concatenate((avg_bad_attrs, [avg_bad_attrs[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, avg_good_attrs, color='blue', linewidth=2, label='Avg Good Image')
    ax.fill(angles, avg_good_attrs, color='blue', alpha=0.25)
    
    ax.plot(angles, avg_bad_attrs, color='red', linewidth=2, label='Avg Bad Image')
    ax.fill(angles, avg_bad_attrs, color='red', alpha=0.25)
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Average Aesthetic Attributes (Good vs Bad)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    
    save_path_radar = os.path.join(figures_dir, "inference_attributes_radar.png")
    plt.savefig(save_path_radar)
    print(f"Saved Attributes Radar chart to {save_path_radar}")

    # 3. Heatmap for Detailed Attributes per Image
    # We want to show: Rows = Images, Cols = Attributes (C, L, F, P, O)
    # Prepare data
    labels = ['Composition (C)', 'Lighting (L)', 'Focus (F)', 'Post-proc (P)', 'Originality (O)']
    row_labels = []
    data_matrix = []
    
    for r in results:
        # Good image row
        row_labels.append(f"{r['set_name']} Good")
        data_matrix.append(r['good_scores'][1:]) # Indices 1-5
        
        # Bad image row
        row_labels.append(f"{r['set_name']} Bad")
        data_matrix.append(r['bad_scores'][1:]) # Indices 1-5
        
    data_matrix = np.array(data_matrix)
    
    fig, ax = plt.subplots(figsize=(10, len(row_labels) * 0.5 + 2))
    im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=1)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(row_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{data_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black")
                           
    ax.set_title("Detailed Attribute Scores per Image")
    fig.tight_layout()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    save_path_heatmap = os.path.join(figures_dir, "inference_attributes_heatmap.png")
    plt.savefig(save_path_heatmap)
    print(f"Saved Detailed Attributes Heatmap to {save_path_heatmap}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = AestheticViT(model_name='vit_tiny_patch16_224', pretrained=False)
    
    model_path = "aesthetic_vit_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Scan validation sets
    base_dir = os.path.dirname(os.path.dirname(__file__)) # aesthetic_project/
    validation_dir = os.path.join(base_dir, "validation")
    
    sets = scan_validation_sets(validation_dir)
    print(f"Found {len(sets)} validation sets.")
    
    results = []
    
    print("\n--- Detailed Inference Results ---\n")
    
    for s in sets:
        print(f"Processing {s['set_name']}...")
        
        good_scores = predict(s['good_path'], model, device)
        bad_scores = predict(s['bad_path'], model, device)
        
        if good_scores is not None and bad_scores is not None:
            results.append({
                "set_name": s['set_name'],
                "good_scores": good_scores,
                "bad_scores": bad_scores
            })
            
            print(f"  {s['set_name']} Good -> IAS: {good_scores[0]:.4f} | C: {good_scores[1]:.2f}, L: {good_scores[2]:.2f}, F: {good_scores[3]:.2f}, O: {good_scores[5]:.2f}")
            print(f"  {s['set_name']} Bad  -> IAS: {bad_scores[0]:.4f} | C: {bad_scores[1]:.2f}, L: {bad_scores[2]:.2f}, F: {bad_scores[3]:.2f}, O: {bad_scores[5]:.2f}")
            print("-" * 40)
            
    # Visualize
    visualize_results(results)

if __name__ == "__main__":
    main()
