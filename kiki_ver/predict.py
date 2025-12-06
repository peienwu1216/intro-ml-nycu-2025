import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train import SegFormer_MTL_Model 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
    print("Success: Using Apple MPS (Metal Performance Shaders) acceleration!")
else:
    device = torch.device("cpu")
    print("Warning: Using CPU. This will be very slow!")
    
MODEL_PATH = "checkpoints/best_model.pth"
TEST_IMG_PATH = "./images/15.jpg" 
TEST_IMG_DIR = 'validation'

# 定義標籤映射 (必須跟訓練時一樣)
SCENE_CLASSES = ["animal", "plant", "human", "static", "architecture", "landscape", "cityscape", "indoor", "night"]
ELEMENT_CLASSES = ["center", "rule_of_thirds", "golden_ratio", "triangle", "horizontal", "vertical", "diagonal", "symmetric", "curved", "radial", "vanishing_point", "pattern", "fill_the_frame"]

print("Loading model...")
model = SegFormer_MTL_Model(num_scenes=len(SCENE_CLASSES), num_elements=len(ELEMENT_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() 

raw_image = cv2.imread(TEST_IMG_PATH)
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
orig_h, orig_w = raw_image.shape[:2]

IMG_SIZE = 384 
input_img = cv2.resize(raw_image, (IMG_SIZE, IMG_SIZE))
vis_img = input_img.copy().astype(np.float32) / 255.0 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(input_img).unsqueeze(0).to(device) 

print("Running inference...")
with torch.no_grad():
    outputs = model(input_tensor)

pred_scores = outputs['scores'][0].cpu().numpy()
scores_dict = {
    "Composition": pred_scores[0],
    "Lighting": pred_scores[1],
    "Focus": pred_scores[2],
    "Originality": pred_scores[3]
}

scene_idx = torch.argmax(outputs['scene'][0]).item()
pred_scene = SCENE_CLASSES[scene_idx]

elem_probs = torch.sigmoid(outputs['elements'][0]).cpu().numpy()
pred_elements = [ELEMENT_CLASSES[i] for i, prob in enumerate(elem_probs) if prob > 0.5]

mask_prob = torch.sigmoid(outputs['mask'][0, 0]).cpu().numpy()
mask_pred = (mask_prob > 0.5).astype(np.uint8) # 二值化
mask_pred_resized = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

class RegressionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['scores']

target_layer = model.unet.encoder.block4[-1] 
cam = GradCAM(model=RegressionWrapper(model), target_layers=[target_layer])
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
gradcam_vis = show_cam_on_image(vis_img, grayscale_cam[0, :], use_rgb=True)
gradcam_vis = cv2.resize(gradcam_vis, (orig_w, orig_h)) 

print("Visualizing...")

mask_overlay = raw_image.copy()
mask_overlay[mask_pred_resized > 0] = [0, 255, 0] 

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(raw_image)
plt.title(f"Scene: {pred_scene}")
plt.axis("off")
info_text = "Scores:\n"
for k, v in scores_dict.items():
    info_text += f"  {k}: {v:.2f}\n"
info_text += "\nElements:\n  " + ", ".join(pred_elements)
plt.text(0, orig_h + 50, info_text, fontsize=12, verticalalignment='top')

plt.subplot(1, 3, 2)
plt.imshow(mask_overlay)
plt.title("Predicted Composition Lines")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gradcam_vis)
plt.title("Grad-CAM (Why high score?)")
plt.axis("off")

plt.tight_layout()
plt.show() 
plt.savefig("result.jpg") 