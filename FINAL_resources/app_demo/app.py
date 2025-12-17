import os
import sys
import time
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from collections import deque

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Path Setup ---
# Add the current directory to sys.path to allow importing from 'field'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from field.config import Config
    from field.model import SwinSAMPNet
    from field.dataset import detect_saliency, IMAGE_NET_MEAN, IMAGE_NET_STD
    from field.utils import dist2ave
except ImportError as e:
    logger.error(f"Failed to import modules from field/: {e}")
    sys.exit(1)

# --- Configuration ---
app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app) # Enable CORS for frontend integration

# Security Config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'heic', 'heif', 'tiff'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Logging Config
LOG_DIR = os.path.join(current_dir, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'app.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Upload Config
UPLOAD_DIR = os.path.join(current_dir, 'uploaded')
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_LOG_FILE = os.path.join(UPLOAD_DIR, 'upload_log.json')

# Rate Limiting (Simple In-Memory)
# Limit: 10 requests per minute per IP
RATE_LIMIT = 10
RATE_WINDOW = 60
request_history = {}

# Model Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
MODEL_PATH = os.path.join(current_dir, 'model', 'best_model.pth')

# --- Global Model Variable ---
model = None
cfg = Config()

# --- Helper Classes & Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_rate_limit(ip):
    current_time = time.time()
    if ip not in request_history:
        request_history[ip] = deque()
    
    history = request_history[ip]
    
    # Remove old requests
    while history and history[0] < current_time - RATE_WINDOW:
        history.popleft()
    
    if len(history) >= RATE_LIMIT:
        return False
    
    history.append(current_time)
    return True

def load_model_once():
    global model
    if model is not None:
        return

    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        # Initialize model
        model = SwinSAMPNet(cfg)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

def apply_colormap_base64(cam, original_image):
    """ Apply colormap and return base64 string """
    # Resize CAM
    orig_size = original_image.size
    cam_resized = cv2.resize(cam, orig_size)
    
    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    orig_np = np.array(original_image)
    overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)
    
    # Convert to Base64
    pil_img = Image.fromarray(overlay)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG", quality=85)
    return base64.b64encode(buff.getvalue()).decode('utf-8')

# --- Routes ---

@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": str(DEVICE)})

import json
import uuid
import datetime

# ... (imports)

def save_upload_info(metadata):
    """Append upload metadata to JSON log file"""
    try:
        log_data = []
        if os.path.exists(UPLOAD_LOG_FILE):
            with open(UPLOAD_LOG_FILE, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        log_data.append(metadata)
        
        with open(UPLOAD_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save upload log: {e}")

# ... (rest of code)

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Rate Limit Check
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    # 2. File Validation
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # 3. Save File & Log
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_uuid = str(uuid.uuid4())
        ext = file.filename.rsplit('.', 1)[1].lower()
        safe_filename = f"{timestamp}_{file_uuid}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Save to disk
        file.save(file_path)
        
        # Log metadata
        metadata = {
            "timestamp": timestamp,
            "uuid": file_uuid,
            "original_filename": secure_filename(file.filename),
            "saved_filename": safe_filename,
            "ip_address": client_ip,
            "user_agent": request.headers.get('User-Agent')
        }
        save_upload_info(metadata)
        logger.info(f"File uploaded and saved: {safe_filename}")

        # Re-open for processing (since stream was consumed by save)
        image = Image.open(file_path).convert('RGB')
        
        # Saliency Detection
        src_np = np.asarray(image).copy()
        sal_map = detect_saliency(src_np, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        sal_tensor = torch.from_numpy(sal_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        ])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 4. Inference & GradCAM
        # Enable gradients for GradCAM
        img_tensor.requires_grad_(True)
        model.zero_grad()
        
        # Forward pass
        pred_dist, pred_attrs = model(img_tensor, sal_tensor)
        
        # Calculate Score for Backward
        weights = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)
        pred_score = (pred_dist * weights).sum()
        
        # Backward pass to populate gradients in model
        pred_score.backward()
        
        # Generate GradCAM using model's internal method
        # The model stores activations and gradients internally during forward/backward
        cam_tensor = model.get_grad_cam(None)
        
        cam = None
        if cam_tensor is not None:
            cam = cam_tensor.squeeze().cpu().detach().numpy()
            
        # Get final score value
        score_val = pred_score.item()
            
        # 5. Format Response
        dist_probs = pred_dist.squeeze().cpu().detach().numpy().tolist()
        attrs = pred_attrs.squeeze().cpu().detach().numpy().tolist()
        
        # Generate GradCAM Base64
        gradcam_b64 = None
        if cam is not None:
            gradcam_b64 = apply_colormap_base64(cam, image)

        response = {
            "score": round(score_val, 4),
            "distribution": dist_probs,
            "attributes": [
                {"name": name, "value": round(val, 4)} 
                for name, val in zip(cfg.ATTRIBUTE_TYPES, attrs)
            ],
            "gradcam_image": f"data:image/jpeg;base64,{gradcam_b64}" if gradcam_b64 else None
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Startup ---
if __name__ == '__main__':
    # Load model at startup
    try:
        load_model_once()
        # Run app
        app.run(host='0.0.0.0', port=5001, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start app: {e}")
