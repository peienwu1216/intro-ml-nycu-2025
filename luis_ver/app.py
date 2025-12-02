import os
import torch
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from model import SwinMTL_NoPost
from dataset import LetterboxPad

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'model'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    
    if not os.path.exists(model_path):
        return None, None

    model = SwinMTL_NoPost()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def predict_image(model, device, image_path):
    transform = transforms.Compose([
        LetterboxPad(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
        return {
            'IAS': outputs['IAS'].item(),
            'C': outputs['C'].item(),
            'L': outputs['L'].item(),
            'F': outputs['F'].item(),
            'O': outputs['O'].item()
        }
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    models = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.pth')]
    models.sort()
    
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        selected_model = request.form.get('model')
        
        if not files or not selected_model:
            return redirect(request.url)
            
        model, device = load_model(selected_model)
        if not model:
            return "Model not found", 404
            
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                scores = predict_image(model, device, filepath)
                if scores:
                    results.append({
                        'filename': filename,
                        'filepath': filepath,
                        'scores': scores
                    })
        
        # Sort results by IAS descending
        results.sort(key=lambda x: x['scores']['IAS'], reverse=True)
        
        return render_template('index.html', models=models, results=results, selected_model=selected_model)

    return render_template('index.html', models=models, results=None)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
