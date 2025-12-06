import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
import timm

# ==========================================
# 1. Dataset Class: deal with images, line drawing, score mapping
# ==========================================
class CADB_MTL_Dataset(Dataset):
    def __init__(self, img_dir, score_path, elem_path, scene_path, attr_path, img_size=384, mode='train'):
        self.img_dir = img_dir
        self.img_size = img_size
        self.mode = mode
        
        with open(score_path) as f: self.scores_data = json.load(f)
        with open(elem_path) as f: self.elem_data = json.load(f)
        with open(scene_path) as f: self.scene_data = json.load(f)
        with open(attr_path) as f: self.attr_data = json.load(f)
        
        # find intersection: ensure images exist in all JSONs(Data Alignment)
        ids_score = set(self.scores_data.keys())
        ids_attr = set(self.attr_data.keys())

        # scene 和 elements 可能有些圖沒有，我們取 score 和 attr 的交集作為主要訓練集
        self.valid_ids = list(ids_score & ids_attr)
        
        # 定義 Scene 類別映射 (需要根據你的 scene_category.json 完整內容填寫)
        self.scene_map = {
            "animal": 0,
            "plant": 1,
            "human": 2,
            "static": 3,
            "architecture": 4,
            "landscape": 5,
            "cityscape": 6,
            "indoor": 7,
            "night": 8
        }
        # 定義 Element 類別映射 (構圖法)
        self.elem_map = {
            "center": 0,
            "rule_of_thirds": 1,
            "golden_ratio": 2,
            "triangle": 3,
            "horizontal": 4,
            "vertical": 5,
            "diagonal": 6,
            "symmetric": 7,  # 或 "symmetry"，看 JSON
            "curved": 8,
            "radial": 9,
            "vanishing_point": 10,
            "pattern": 11,
            "fill_the_frame": 12
        }

    def _get_scores(self, img_id):
        """
        將 AADB Attributes 和 CADB Scores 映射到 C, L, F, O 四個分數
        """
        # 1. Composition (C): 來自 CADB 的 mean score (1-5 -> normalize to 0-1)
        c_raw = self.scores_data[img_id]['mean']
        c_score = (c_raw - 1.0) / 4.0
        
        # 取得屬性 (AADB 通常是 -1 到 1，或者是 0 到 1，這裡假設已經是 0-1 或需要標準化)
        # 假設 AADB 屬性值在 JSON 裡是 -1.0 ~ 1.0，我們將其轉為 0.0 ~ 1.0
        attrs = self.attr_data[img_id]
        def norm_attr(val): return (val + 1.0) / 2.0 

        # 2. Light/Color (L)
        l_score = (norm_attr(attrs.get('Light', 0)) * 0.4 + 
                   norm_attr(attrs.get('ColorHarmony', 0)) * 0.3 + 
                   norm_attr(attrs.get('VividColor', 0)) * 0.3)

        # 3. Focus/Clarity (F)
        f_score = (norm_attr(attrs.get('DoF', 0)) * 0.5 + 
                   norm_attr(attrs.get('Object', 0)) * 0.5)

        # 4. Originality/Story (O)
        o_score = norm_attr(attrs.get('Content', 0))

        return torch.tensor([c_score, l_score, f_score, o_score], dtype=torch.float32)
    
    

    def _draw_mask(self, img_shape, img_id):
        """
        根據 JSON 座標畫出構圖線 Mask (兼容直線、多邊形、曲線、點)
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if img_id in self.elem_data:
            elements = self.elem_data[img_id]
            
            thickness = max(5, min(h, w) // 100)
            
            for elem_type, lines in elements.items():
                for line in lines:
                    coords = list(map(int, line))
                    num_coords = len(coords)

                    # Case A: 單點 (Vanishing Point) - 2個數值
                    if num_coords == 2:
                        center = (coords[0], coords[1])
                        # 畫一個實心圓點，半徑設大一點
                        cv2.circle(mask, center, thickness * 2, 255, -1)

                    # Case B: 直線 (Vertical, Horizontal, etc.) - 4個數值
                    elif num_coords == 4:
                        pt1 = (coords[0], coords[1])
                        pt2 = (coords[2], coords[3])
                        cv2.line(mask, pt1, pt2, 255, thickness)

                    # Case C: 多邊形/折線 (Triangle, Curved, Pattern) - >4個數值
                    elif num_coords > 4:
                        # 將座標 reshape 成 (N, 1, 2) 的格式供 polylines 使用
                        # numpy array shape: (點的數量, 1, 2)
                        pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                        
                        # 判斷是否要封閉圖形
                        # Triangle, Symmetry, Pattern 通常是封閉的
                        # Curved, Diagonal 可能是折線
                        is_closed = elem_type in ["triangle", "pattern", "symmetry", "symmetric"]
                        
                        cv2.polylines(mask, [pts], is_closed, 255, thickness)
        
        return mask

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        
        # 1. 讀取圖片
        image = cv2.imread(img_path)
        if image is None: # 錯誤處理
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.img_size, self.img_size
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

        # 2. 產生 Ground Truth Mask (在原始尺寸畫，再縮放，確保位置精準)
        mask = self._draw_mask((orig_h, orig_w), img_id)

        # 3. Resize (圖片與 Mask 同步縮放)
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 4. 取得其他標籤
        scores = self._get_scores(img_id) # (4,)
        
        # Scene Label (One-hot index)
        scene_str = self.scene_data.get(img_id, "unknown")
        scene_label = self.scene_map.get(scene_str, 0) # 預設 0
        
        # Element Label (Multi-hot)
        elem_vec = torch.zeros(len(self.elem_map))
        if img_id in self.elem_data:
            for k in self.elem_data[img_id].keys():
                if k in self.elem_map:
                    elem_vec[self.elem_map[k]] = 1.0

        # 5. To Tensor & Normalize

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 
    
        # 修正點：使用 torchvision.transforms.Normalize
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])
        image = normalizer(image)
        
        mask = torch.from_numpy(mask).float() / 255.0 
        mask = mask.unsqueeze(0) # (1, H, W)

        return {
            'image': image,
            'mask': mask,
            'scores': scores,
            'scene': torch.tensor(scene_label, dtype=torch.long),
            'elements': elem_vec
        }

    def __len__(self):
        return len(self.valid_ids)

class SafeFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(SafeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # 關鍵修正：確保 inputs 和 targets 都在同一個 device，且為 float
        if targets.device != inputs.device:
            targets = targets.to(inputs.device)
        if targets.dtype != inputs.dtype:
            targets = targets.float()

        # 計算 BCE Loss
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# ==========================================
# 2. Model: SegFormer (MiT) + U-Net Decoder + Heads
# ==========================================
class SegFormer_MTL_Model(nn.Module):
    def __init__(self, num_scenes, num_elements, num_scores=4):
        super().__init__()
        
        self.unet = smp.Unet(
            encoder_name="mit_b3",        # 錯誤訊息中支援的 encoder
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,                    # Mask 輸出通道數
        )
        
        self.encoder_channels = self.unet.encoder.out_channels[-1]
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # --- Task Heads (完全不用動) ---
        # 1. Scoring Head (Regression 0-1)
        self.head_score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_channels, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_scores),
            nn.Sigmoid() 
        )
        
        # 2. Scene Classification Head
        self.head_scene = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_channels, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_scenes)
        )
        
        # 3. Element Classification Head (Multi-label)
        self.head_element = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder_channels, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_elements)
        )

    def forward(self, x):
        # 1. Encoder Pass (提取特徵)
        features = self.unet.encoder(x)
        
        # 2. Decoder Pass (畫線任務)
        # ---------------------------------------------------------
        decoder_output = self.unet.decoder(features)
        # ---------------------------------------------------------
        
        mask_pred = self.unet.segmentation_head(decoder_output)
        
        # 3. Heads Pass (分類/評分任務)
        # SegFormer 的 features 是一個 list，最後一層特徵在最後面
        last_feat = features[-1]
        pooled_feat = self.pool(last_feat)
        
        score_pred = self.head_score(pooled_feat)
        scene_pred = self.head_scene(pooled_feat)
        element_pred = self.head_element(pooled_feat)
        
        return {
            'mask': mask_pred,
            'scores': score_pred,
            'scene': scene_pred,
            'elements': element_pred,
            'last_feat': last_feat 
        }


# ==========================================
# 3. Loss Function: 權重平衡
# ==========================================
class MTL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # Mask Loss: Dice (結構) + Focal (類別不平衡)
        self.loss_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        # self.loss_focal = smp.losses.FocalLoss(mode='binary')
        self.loss_focal = SafeFocalLoss()
        
        # Score Loss: MSE
        self.loss_score = nn.MSELoss()
        
        # Scene Loss: Cross Entropy
        self.loss_scene = nn.CrossEntropyLoss()
        
        # Element Loss: BCE (Multi-label)
        self.loss_element = nn.BCEWithLogitsLoss()
        
        # Weights (需要根據實驗調整)
        self.w = {'mask': 10.0, 'score': 20.0, 'scene': 1.0, 'elements': 1.0}

    def forward(self, preds, targets):
        # Mask Loss
        l_dice = self.loss_dice(preds['mask'], targets['mask'])
        l_focal = self.loss_focal(preds['mask'], targets['mask'])
        l_mask = l_dice + l_focal
        
        # Others
        l_score = self.loss_score(preds['scores'], targets['scores'])
        l_scene = self.loss_scene(preds['scene'], targets['scene'])
        l_element = self.loss_element(preds['elements'], targets['elements'])
        
        total_loss = (self.w['mask'] * l_mask + 
                      self.w['score'] * l_score + 
                      self.w['scene'] * l_scene + 
                      self.w['elements'] * l_element)
        
        return total_loss, {
            'l_mask': l_mask.item(), 
            'l_score': l_score.item(), 
            'l_scene': l_scene.item()
        }

# ==========================================
# 4. Grad-CAM Helper
# ==========================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_gradcam(model, input_tensor, target_layer, rgb_img):
    """
    input_tensor: (1, 3, H, W) normalized
    rgb_img: (H, W, 3) float 0-1 (for visualization overlay)
    """
    class RegressionWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            # 只回傳 scores output
            return self.model(x)['scores']

    wrapped_model = RegressionWrapper(model)
    
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    
    targets = [ClassifierOutputTarget(0)] 

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
    return visualization

# ==========================================
# Main Execution Example
# ==========================================
if __name__ == "__main__":

    IMG_DIR = "./images"
    JSON_SCORE = "./composition_scores.json"
    JSON_ELEM = "./composition_elements.json"
    JSON_SCENE = "./scene_categories.json"
    JSON_ATTR = "./composition_attributes.json"

    
    dataset = CADB_MTL_Dataset(IMG_DIR, JSON_SCORE, JSON_ELEM, JSON_SCENE, JSON_ATTR)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # 檢查 Mac GPU
        device = torch.device("mps")
        print("Success: Using Apple MPS (Metal Performance Shaders) acceleration!")
    else:
        device = torch.device("cpu")
        print("Warning: Using CPU. This will be very slow!")
    model = SegFormer_MTL_Model(num_scenes=9, num_elements=13)
    model.to(device)

    # 3. Loss & Optimizer
    criterion = MTL_Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    
    num_epochs = 10 
    
    best_loss = float('inf') 

    print("Start Training Loop...")

    for epoch in range(num_epochs):
        model.train() # 確保模型在訓練模式
        running_loss = 0.0 # 用來計算這個 Epoch 的平均 Loss
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            batch['mask'] = batch['mask'].to(device)
            batch['scores'] = batch['scores'].to(device)
            batch['scene'] = batch['scene'].to(device)
            batch['elements'] = batch['elements'].to(device)
            
            # Forward
            preds = model(images)
            
            # Calculate Loss
            loss, loss_dict = criterion(preds, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Finished. Average Loss: {epoch_avg_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(model.state_dict(), "checkpoints/latest_model.pth")
        
        if epoch_avg_loss < best_loss:
            print(f"Loss improved from {best_loss:.4f} to {epoch_avg_loss:.4f}. Saving best model...")
            best_loss = epoch_avg_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        
        print("-" * 30)