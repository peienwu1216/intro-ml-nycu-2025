import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from torchsummary import summary

# Import definitions from main
from main import ImprovedCNN, BaselineCNN, BaselineNN, FashionMNISTDataset, train_one_epoch, validate_one_epoch, set_seed

# Set device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    
print(f"Using Device: {DEVICE}")

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_confusion_matrix(model, val_loader, output_dir='output'):
    print("\n📊 Generating Confusion Matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Improved CNN)')
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"✅ Confusion matrix saved to {output_dir}/confusion_matrix.png")

def train_baseline_nn(train_loader, val_loader, output_dir='output', epochs=20):
    print("\n🧠 Training Baseline NN for Comparison...")
    # Use the correct BaselineNN class from main
    model = BaselineNN().to(DEVICE)
    
    # Use same optimizer setup as CNN for fair comparison, maybe simpler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] Val Acc: {val_acc:.2f}%")
            
    # Plot Baseline NN curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Baseline NN Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Baseline NN Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_nn_learning_curves.png'))
    plt.close()
    
    # Save Params count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Baseline NN trained. Best Val Acc: {max(history['val_acc']):.2f}%. Params: {params:,}")
    return params, max(history['val_acc'])

def print_model_summary(model, input_size=(1, 28, 28)):
    print("\n📝 Model Architecture Summary (for Layer Table):")
    # Use torchsummary if available, else print standard
    try:
        summary(model, input_size)
    except Exception as e:
        print("torchsummary not found or failed. Using default print.")
        print(model)
        
    print("\n--- Layer Detail Helper for Hand Calculation ---")
    # Manually iterate to show shapes useful for formula: (C_in * kH * kW + 1) * C_out
    # This is a rough helper, manual verification is needed against code structure.
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Layer: {name} | Type: Conv2d | In: {module.in_channels} | Out: {module.out_channels} | Kernel: {module.kernel_size} | Stride: {module.stride} | Padding: {module.padding} | Bias: {module.bias is not None}")
        elif isinstance(module, nn.Linear):
            print(f"Layer: {name} | Type: Linear | In: {module.in_features} | Out: {module.out_features} | Bias: {module.bias is not None}")
        elif isinstance(module, (nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
             print(f"Layer: {name} | Type: {type(module).__name__}")

def main():
    set_seed(42)
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    print("Loading Data...")
    # Use standard transform for validation/analysis
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = FashionMNISTDataset("data/train.csv", mode='train', model_type='cnn', transform=transform)
    val_size = int(0.1 * len(full_dataset)) # 10% split for consistency with previous runs
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_set = Subset(full_dataset, val_indices.indices)
    train_set = Subset(full_dataset, train_indices.indices)
    
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    # 2. Improved CNN Analysis
    print("\n🔍 Analyzing Improved CNN...")
    improved_model = ImprovedCNN(use_bn=True, use_dropout=True, use_residual=True, pooling_type='max').to(DEVICE)
    
    # Try to load best weights if available
    if os.path.exists("best_model_fashion.pth"):
        try:
            improved_model.load_state_dict(torch.load("best_model_fashion.pth", map_location=DEVICE))
            print("Loaded weights from best_model_fashion.pth")
        except:
            print("⚠️ Warning: Could not load best_model_fashion.pth. Using random weights (Confusion Matrix will be random!).")
            print("   Please ensure you have run main.py training first.")
    
    # 2a. Confusion Matrix
    plot_confusion_matrix(improved_model, val_loader, output_dir)
    
    # 2b. Layer Details
    print_model_summary(improved_model)
    
    # 3. Baseline NN Analysis
    # BaselineNN is imported from main and handles flattening in its forward method.
    # FashionMNISTDataset with model_type='cnn' returns (1, 28, 28), 
    # so the model's internal flatten will convert it to (784).
    
    train_baseline_nn(train_loader, val_loader, output_dir)
    
    print(f"\n✨ All Analysis Completed! Check '{output_dir}' folder for:")
    print("1. confusion_matrix.png")
    print("2. baseline_nn_learning_curves.png")
    print("3. Console output for Layer Table data")

if __name__ == "__main__":
    try:
        import torchsummary
    except ImportError:
        print("Installing torchsummary for layer analysis...")
        os.system("pip install torchsummary")
        
    main()

