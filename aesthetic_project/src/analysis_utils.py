import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageStat
import io

def get_dominant_colors(image, num_colors=5):
    """
    Extract dominant colors using PIL's quantization.
    Returns a list of (color_tuple, count) sorted by count desc.
    """
    # Resize to speed up
    img_small = image.copy()
    img_small.thumbnail((150, 150))
    
    # Quantize to num_colors
    result = img_small.quantize(colors=num_colors)
    
    # Get palette
    palette = result.getpalette()
    color_counts = result.getcolors()
    
    # color_counts is list of (count, index)
    # palette is list of [r, g, b, r, g, b, ...]
    
    dominant_colors = []
    total_pixels = sum(count for count, idx in color_counts)
    
    for count, idx in color_counts:
        r = palette[idx*3]
        g = palette[idx*3+1]
        b = palette[idx*3+2]
        dominant_colors.append(((r, g, b), count))
        
    # Sort by count descending
    dominant_colors.sort(key=lambda x: x[1], reverse=True)
    
    return dominant_colors

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def plot_color_palette(dominant_colors):
    """
    Creates a matplotlib figure of the color palette.
    """
    colors = [c[0] for c in dominant_colors]
    counts = [c[1] for c in dominant_colors]
    total = sum(counts)
    proportions = [c/total for c in counts]
    
    # Normalized RGB for matplotlib
    colors_norm = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]
    
    fig, ax = plt.subplots(figsize=(6, 2))
    current_pos = 0
    
    for i, (color, prop) in enumerate(zip(colors_norm, proportions)):
        ax.barh(0, prop, left=current_pos, color=color, height=1, edgecolor='none')
        # Add text if proportion is big enough
        if prop > 0.1:
            hex_code = rgb_to_hex((int(color[0]*255), int(color[1]*255), int(color[2]*255)))
            # Choose text color based on brightness
            text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
            ax.text(current_pos + prop/2, 0, hex_code, ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')
            
        current_pos += prop
        
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    plt.title("Dominant Color Palette", fontsize=10)
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def analyze_technical_stats(image):
    """
    Calculate Brightness, Contrast, and Sharpness.
    """
    # Convert to cv2 format (numpy array)
    img_np = np.array(image)
    if img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
        
    # Brightness (Mean)
    brightness = np.mean(img_gray)
    
    # Contrast (Std Dev)
    contrast = np.std(img_gray)
    
    # Sharpness (Laplacian Variance)
    sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    return {
        "Brightness": brightness,
        "Contrast": contrast,
        "Sharpness": sharpness
    }

