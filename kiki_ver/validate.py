import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import sys

from train import SegFormer_MTL_Model 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= Configuration =================
MODEL_PATH = "checkpoints/best_model.pth"
VALIDATION_DIR = "validation"  
OUTPUT_DIR = "validation_outputs" 
RESULT_TXT = "validation_results.txt" 

SCENE_CLASSES = ["animal", "plant", "human", "static", "architecture", "landscape", "cityscape", "indoor", "night"]
ELEMENT_CLASSES = ["center", "rule_of_thirds", "golden_ratio", "triangle", "horizontal", "vertical", "diagonal", "symmetric", "curved", "radial", "vanishing_point", "pattern", "fill_the_frame"]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ================= Helper Classes & Functions =================

class RegressionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['scores']

def generate_feedback(scores):
    suggestions = []
    LOW_THRESH = 0.45
    HIGH_THRESH = 0.75
    if scores['Composition'] < LOW_THRESH:
        suggestions.append("Composition is weak. Try Rule of Thirds.")
    elif scores['Composition'] > HIGH_THRESH:
        suggestions.append("Strong composition structure detected.")
    if scores['Lighting'] < LOW_THRESH:
        suggestions.append("Lighting is flat/dim.")
    if scores['Focus'] < LOW_THRESH:
        suggestions.append("Subject lacks sharpness.")
    if scores['Originality'] > HIGH_THRESH:
        suggestions.append("Very creative!")
    if not suggestions:
        suggestions.append("Technically balanced but could be bolder.")
    return suggestions

def process_single_image(model, img_path, img_size=384):
    raw_image = cv2.imread(img_path)
    if raw_image is None: return None
    
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = raw_image.shape[:2]

    input_img = cv2.resize(raw_image, (img_size, img_size))
    vis_img = input_img.copy().astype(np.float32) / 255.0 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    pred_scores = outputs['scores'][0].cpu().numpy()
    overall_score = np.mean(pred_scores)
    
    scores_dict = {
        "Composition": pred_scores[0],
        "Lighting": pred_scores[1],
        "Focus": pred_scores[2],
        "Originality": pred_scores[3],
        "Overall": overall_score
    }

    scene_idx = torch.argmax(outputs['scene'][0]).item()
    pred_scene = SCENE_CLASSES[scene_idx]

    elem_probs = torch.sigmoid(outputs['elements'][0]).cpu().numpy()
    pred_elements = [ELEMENT_CLASSES[i] for i, prob in enumerate(elem_probs) if prob > 0.5]

    mask_prob = torch.sigmoid(outputs['mask'][0, 0]).cpu().numpy()
    mask_pred = (mask_prob > 0.5).astype(np.uint8)
    mask_pred_resized = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    target_layer = model.unet.encoder.block4[-1]
    cam = GradCAM(model=RegressionWrapper(model), target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
    gradcam_vis = show_cam_on_image(vis_img, grayscale_cam[0, :], use_rgb=True)
    gradcam_vis = cv2.resize(gradcam_vis, (orig_w, orig_h))

    return {
        "raw_image": raw_image,
        "scores": scores_dict,
        "scene": pred_scene,
        "elements": pred_elements,
        "mask": mask_pred_resized,
        "gradcam": gradcam_vis,
        "filename": os.path.basename(img_path)
    }

def save_visual_result(result_data, save_path, feedback):
    # (保持不變)
    raw_image = result_data['raw_image']
    orig_h, _ = raw_image.shape[:2]
    mask_overlay = raw_image.copy()
    mask_overlay[result_data['mask'] > 0] = [0, 255, 0] 

    plt.figure(figsize=(18, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_image)
    plt.title(f"Scene: {result_data['scene']}")
    plt.axis("off")
    
    info_text = "Scores:\n"
    for k in ["Composition", "Lighting", "Focus", "Originality"]:
        v = result_data['scores'][k]
        info_text += f"  {k}: {v:.2f}\n"
    info_text += f"  Overall: {result_data['scores']['Overall']:.2f}\n"

    info_text += "\nElements:\n  " + ", ".join(result_data['elements'])
    info_text += "\n\nFeedback:\n" + "\n".join([f"- {s}" for s in feedback])
    
    plt.text(0, orig_h + 50, info_text, fontsize=11, verticalalignment='top')
    plt.subplot(1, 3, 2)
    plt.imshow(mask_overlay)
    plt.title("Predicted Lines")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(result_data['gradcam'])
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_global_grouped_bar_chart(stats_data, save_path):
    """
    stats_data: dict, key=set_name, value={'good': [scores...], 'bad': [scores...]}
    """
    valid_sets = [k for k in stats_data.keys() if stats_data[k]['good'] or stats_data[k]['bad']]
    
    # 排序 set1, set2...
    valid_sets.sort(key=lambda x: int(x.replace('set', '')) if x.replace('set', '').isdigit() else x)
    
    if not valid_sets:
        print("No valid data for grouped bar chart.")
        return

    means_good = []
    means_bad = []
    
    for s in valid_sets:
        g_scores = stats_data[s]['good']
        b_scores = stats_data[s]['bad']
        
        means_good.append(np.mean(g_scores) if g_scores else 0)
        means_bad.append(np.mean(b_scores) if b_scores else 0)

    x = np.arange(len(valid_sets))
    width = 0.35 

    plt.figure(figsize=(12, 6))
    
    rects1 = plt.bar(x - width/2, means_good, width, label='Good', color='#77dd77', edgecolor='white') # 綠色
    rects2 = plt.bar(x + width/2, means_bad, width, label='Bad', color='#ff6961', edgecolor='white')   # 紅色

    plt.ylabel('Average Overall Score')
    plt.title('Performance Comparison: Good vs Bad Images by Set')
    plt.xticks(x, valid_sets)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 在柱子上標示數值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                         f'{height:.2f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Global Plot] Saved Good/Bad comparison chart to {save_path}")

# ================= Main Loop =================

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = SegFormer_MTL_Model(num_scenes=len(SCENE_CLASSES), num_elements=len(ELEMENT_CLASSES))
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_TXT, "w") as f:
        f.write("Validation Results\n==================\n")

    global_stats = {}

    for set_num in range(1, 11):
        set_folder_name = f"set{set_num}"
        base_folder = os.path.join(VALIDATION_DIR, set_folder_name)
        
        global_stats[set_folder_name] = {'good': [], 'bad': []}

        if not os.path.isdir(base_folder):
            continue

        print(f"\nProcessing Folder: {base_folder}")
        
        for root, dirs, files in os.walk(base_folder):
            for img_name in files:
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(root, img_name)
                
                path_lower = img_path.lower()
                category = "unknown"
                
                if "good" in path_lower:
                    category = "good"
                elif "bad" in path_lower:
                    category = "bad"
                else:
                    pass 

                print(f"  Testing {img_name} (Class: {category})...")
                
                try:
                    result = process_single_image(model, img_path)
                    if result is None: continue

                    if category in ['good', 'bad']:
                        global_stats[set_folder_name][category].append(result['scores']['Overall'])
                    feedback = generate_feedback(result['scores'])

                    # Save a single canonical file per set/category to avoid duplicates.
                    # Target path: OUTPUT_DIR/setX/<category>.jpg  (e.g. output/set1/good.jpg)
                    set_output_dir = os.path.join(OUTPUT_DIR, set_folder_name)
                    os.makedirs(set_output_dir, exist_ok=True)
                    save_path = os.path.join(set_output_dir, f"{category}.jpg")
                    save_visual_result(result, save_path, feedback)

                    rel_path = os.path.relpath(save_path)
                    with open(RESULT_TXT, "a") as f:
                        f.write(f"File: {rel_path} | Class: {category} | Overall: {result['scores']['Overall']:.2f}\n")

                except Exception as e:
                    print(f"    -> Error: {e}")

    print("\nGenerating Global Good/Bad Comparison Chart...")
    chart_path = os.path.join(OUTPUT_DIR, "GLOBAL_GOOD_VS_BAD_CHART.jpg")
    plot_global_grouped_bar_chart(global_stats, chart_path)

    print(f"\nDone! Check {OUTPUT_DIR}/GLOBAL_GOOD_VS_BAD_CHART.jpg")