import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AADBDataset(Dataset):
    def __init__(self, label_dir, img_dir, split='Train', transform=None):
        """
        label_dir: Folder containing the attribute txt files
        img_dir: Folder containing images
        split: 'Train' or 'Validation' or 'TestNew'
        """
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.image_data = {} # {filename: {attr: value}}
        self.filenames = []

        # Map internal attribute names to file suffixes
        # Note: 'BalacingElements' has a typo in the filename in the dataset
        self.attr_file_map = {
            'score': 'score',
            'balancing': 'BalacingElements', 
            'color_harmony': 'ColorHarmony',
            'content': 'Content',
            'dof': 'DoF',
            'light': 'Light',
            'motion_blur': 'MotionBlur',
            'object': 'Object',
            'repetition': 'Repetition',
            'rule_of_thirds': 'RuleOfThirds',
            'symmetry': 'Symmetry',
            'vivid_color': 'VividColor'
        }

        # Read 'score' first to initialize filenames
        score_file = os.path.join(label_dir, f"imgList{split}Regression_score.txt")
        if not os.path.exists(score_file):
             raise FileNotFoundError(f"Score file not found: {score_file}")
             
        with open(score_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    val = float(parts[1])
                    self.filenames.append(img_name)
                    self.image_data[img_name] = {'score': val}

        # Read other attributes
        for attr_key, file_suffix in self.attr_file_map.items():
            if attr_key == 'score': continue
            
            file_path = os.path.join(label_dir, f"imgList{split}Regression_{file_suffix}.txt")
            if not os.path.exists(file_path):
                print(f"Warning: Attribute file not found: {file_path}")
                # Fill with 0.5 as default if missing? Or raise error?
                # For now, let's set to 0.5 to avoid crash, but warn.
                for img_name in self.filenames:
                    self.image_data[img_name][attr_key] = 0.5
                continue
                
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        val = float(parts[1])
                        if img_name in self.image_data:
                            # Normalize attributes from [-1, 1] to [0, 1]
                            # Assumption based on AADB data format where attributes are -1 to 1
                            # Score is already 0-1
                            val = (val + 1.0) / 2.0
                            self.image_data[img_name][attr_key] = val

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        attrs = self.image_data[img_name]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(f"Warning: Could not open {img_path}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # --- Label Mapping ---
        # 1. Light/Color (L)
        # Avg(good_lighting, color_harmony, vivid_color)
        score_L = (attrs.get('light', 0.5) + attrs.get('color_harmony', 0.5) + attrs.get('vivid_color', 0.5)) / 3.0
        
        # 2. Composition (C)
        # Avg(rule_of_thirds, balancing_element, symmetry, repetition)
        score_C = (attrs.get('rule_of_thirds', 0.5) + attrs.get('balancing', 0.5) + 
                   attrs.get('symmetry', 0.5) + attrs.get('repetition', 0.5)) / 4.0
        
        # 3. Focus/Clarity (F)
        # Avg(shallow_dof, object_emphasis, 1 - motion_blur)
        # Note: Proposal says Focus/Clarity = Avg(Shallow DOF, Object Emphasis, 1 - Motion Blur) ?
        # Let's double check the prompt logic: "Focus/Clarity (F) <- Avg(shallow_depth_of_field, object_emphasis, 1 - motion_blur)"
        score_F = (attrs.get('dof', 0.5) + attrs.get('object', 0.5) + (1.0 - attrs.get('motion_blur', 0.5))) / 3.0
        
        # 4. Originality/Story (O)
        # interesting_content
        score_O = attrs.get('content', 0.5)
        
        # 5. Post-processing (P)
        # Set as 0.5 constant (as per prompt instruction)
        score_P = 0.5
        
        # IAS (Total Score)
        raw_score = attrs.get('score', 0.5)
        
        # Target Tensor: [Global, C, L, F, P, O]
        targets = torch.tensor([raw_score, score_C, score_L, score_F, score_P, score_O], dtype=torch.float32)
        
        return image, targets
