import gradio as gr
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from model import AestheticViT, AestheticSwin
from config import Config
import io
import os

# Import analysis utils
from analysis_utils import get_dominant_colors, plot_color_palette, analyze_technical_stats

# 1. 載入模型
# Get the directory of the current script to construct absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "aesthetic_vit_model.pth")

device = torch.device("cpu") # Demo 用 CPU 即可

# Check if model exists
if not os.path.exists(MODEL_PATH):
    # Fallback to swin model name if vit not found, or generic name
    MODEL_PATH = os.path.join(BASE_DIR, "aesthetic_vit_model.pth") 
    if not os.path.exists(MODEL_PATH):
         # Try current default model name if different
         pass

# Initialize model based on Config
if 'swin' in Config.MODEL_NAME:
    model = AestheticSwin(model_name=Config.MODEL_NAME, pretrained=False)
else:
    model = AestheticViT(model_name=Config.MODEL_NAME, pretrained=False)

# Load with weights_only=True for security if possible, but for now stick to default or safe usage
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
else:
    print(f"Warning: Model file {MODEL_PATH} not found. Using random weights.")

model.to(device)
model.eval()

# 2. 預處理函數
# Same as validation transform in train.py
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定義 4 個維度的建議模板 (移除 Post-proc)
ADVICE_DB = {
    "Composition": {
        "low": "⚠️ 構圖稍顯雜亂。建議嘗試「三分法」將主體放在交叉點，或尋找畫面中的引導線來突出重點。",
        "high": "✅ 構圖非常穩重！主體位置安排得宜，視覺平衡感很好。"
    },
    "Light": {
        "low": "⚠️ 光線似乎過暗或過曝，導致細節流失。建議調整曝光補償，或利用側光來增加立體感。",
        "high": "✅ 光影運用出色！色彩與光線的搭配讓畫面很有氛圍。"
    },
    "Focus": {
        "low": "⚠️ 主體看起來有點模糊。建議提高快門速度防止手震，或確認對焦點是否準確落在主體上。",
        "high": "✅ 清晰度極佳！主體銳利，景深控制得宜，成功突顯了重點。"
    },
    "Originality": {
        "low": "💡 題材較為常見。試著改變拍攝視角（如低角度或俯拍），或尋找獨特的時間點來增加故事性。",
        "high": "✅ 非常獨特的視角！這張照片展現了與眾不同的創意與故事性。"
    }
}

def generate_diagnosis(sub_scores):
    """
    根據子分數生成診斷報告
    sub_scores 順序: [Composition, Light, Focus, Originality]
    """
    labels = ['Composition', 'Light', 'Focus', 'Originality']
    
    # 找出最高分與最低分的項目
    max_idx = np.argmax(sub_scores)
    min_idx = np.argmin(sub_scores)
    
    max_score = sub_scores[max_idx]
    min_score = sub_scores[min_idx]
    
    label_best = labels[max_idx]
    label_worst = labels[min_idx]
    
    report = []
    
    # 1. 讚美優點 (如果最高分大於 0.6)
    if max_score > 0.6:
        report.append(f"【優點】{ADVICE_DB[label_best]['high']}")
    
    # 2. 改進建議 (針對最低分項目)
    # 如果最低分真的很低 (< 0.5)，給出具體建議
    if min_score < 0.5:
        report.append(f"【建議】{ADVICE_DB[label_worst]['low']}")
    else:
        # 如果連最低分都很高，給予通用讚美
        report.append("🎉 這張照片各方面表現都很均衡，是一張優秀的作品！")
        
    return "\n".join(report)

def predict(image):
    if image is None:
        return "Please upload an image.", None, None, "No image uploaded."

    # --- Model Prediction ---
    # 預處理
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor).squeeze().numpy()
        
    # 數值限制在 0-1 之間 (避免顯示爆掉)
    outputs = np.clip(outputs, 0, 1)
    
    # 解析分數
    ias = outputs[0]
    sub_scores = outputs[1:]
    labels = ['Composition', 'Light', 'Focus', 'Originality']
    
    # --- Analysis: Color Palette Only ---
    dominant_colors = get_dominant_colors(image)
    palette_img = plot_color_palette(dominant_colors)
    
    # --- 繪製雷達圖 ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = sub_scores.tolist()
    # 閉合圖形
    values += [values[0]]
    angles += [angles[0]]
    labels_plot = labels + [labels[0]]
    
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    
    # Use fixed scale 0-1 for consistency
    ax.set_ylim(0, 1)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"Total Aesthetic Score: {ias:.2f}/1.0", size=15, color='blue', y=1.1)
    
    # 將圖表轉為圖片回傳
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close(fig)
    
    # --- 生成文字建議 ---
    # 純根據 ML 模型預測結果 (sub_scores) 生成
    diagnosis_text = generate_diagnosis(sub_scores)
    
    # 回傳：分數字串, 雷達圖, 調色盤, 文字建議
    final_score_str = f"Score: {ias*10:.1f} / 10"
    return final_score_str, plot_img, palette_img, diagnosis_text

# 3. 建立 Gradio 介面
# Find a valid example image
example_path = os.path.join(BASE_DIR, "../validation/set1/good.jpg")
examples = [example_path] if os.path.exists(example_path) else []

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Photo"),
    outputs=[
        gr.Label(label="Aesthetic Score"), 
        gr.Image(type="pil", label="Radar Chart Analysis"),
        gr.Image(type="pil", label="Dominant Color Palette"),
        gr.Textbox(label="AI Diagnosis & Suggestions", lines=10)
    ],
    title="📸 AI Aesthetic Scorer",
    description="上傳照片，AI 將分析構圖、光線、清晰度等 4 大美學維度，並提供色彩與技術指標分析。",
    examples=examples
)

if __name__ == "__main__":
    print("Starting Gradio app...")
    iface.launch(share=True) # share=True 會生成一個公開連結
