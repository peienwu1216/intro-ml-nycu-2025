import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Import from main, which now contains all definitions
from main import ImprovedCNN, ImprovedNN, FashionMNISTDataset, train_one_epoch, validate_one_epoch, set_seed

# Set device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using Device: {DEVICE}")

def run_experiment(config_name, model_cls, model_params, train_loader, val_loader, epochs=25):
    """
    Run a single experiment with specific model parameters.
    """
    print(f"\n🧪 Starting Experiment: {config_name}")
    print(f"   Params: {model_params}")
    
    # 1. Initialize Model
    model = model_cls(**model_params).to(DEVICE)
    
    # 2. Count Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters: {total_params:,}")
    
    # 3. Setup Training
    # Using same settings as main.py but slightly simpler for ablation speed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # 4. Training Loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Simple progress log every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] Val Acc: {val_acc:.2f}%")
            
    print(f"✅ Experiment {config_name} Completed. Best Val Acc: {best_val_acc:.2f}%")
    
    return {
        'config': config_name,
        'params': total_params,
        'best_val_acc': best_val_acc,
        'history': history
    }

def plot_results(results, output_dir='output'):
    """
    Plot comparison curves for Loss and Accuracy.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Accuracy Plot
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['history']['val_acc'], label=f"{name} (Best: {res['best_val_acc']:.1f}%)")
    
    plt.title("Ablation Study: Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ablation_val_accuracy.png"))
    plt.close()
    
    # 2. Loss Plot
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        plt.plot(res['history']['val_loss'], label=name)
        
    plt.title("Ablation Study: Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ablation_val_loss.png"))
    plt.close()

def save_summary_table(results, output_dir='output'):
    """
    Save a CSV summary of the experiments.
    """
    data = []
    for name, res in results.items():
        data.append({
            'Model Variant': name,
            'Parameters': res['params'],
            'Best Val Accuracy (%)': f"{res['best_val_acc']:.2f}",
            'Epochs': 25
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📄 Summary table saved to {csv_path}")
    print(df)

def main():
    set_seed(42)
    
    # 1. Prepare Data
    print("Loading Data...")
    
    # Transforms (Simple ones for ablation speed/stability comparison, no need for heavy aug here)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = FashionMNISTDataset("data/train.csv", mode='train', model_type='cnn', transform=transform)
    
    # Use fixed split for fair comparison
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    # Create fixed indices for reproducibility
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_set = Subset(full_dataset, train_indices.indices)
    val_set = Subset(full_dataset, val_indices.indices)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)
    
    # 2. Define Experiments (Ablation Studies)
    # Mandatory Experiments from Spec:
    # For Both NN and CNN:
    # 1. BN: With vs Without
    # 2. Dropout: With vs Without
    # For CNN only:
    # 3. Residual: With vs Without
    # 4. Pooling: Max vs Stride
    
    experiments_list = [
        # --- CNN Ablations ---
        {"name": "CNN_Full",       "cls": ImprovedCNN, "params": {"use_bn": True,  "use_dropout": True,  "use_residual": True,  "pooling_type": 'max'}},
        {"name": "CNN_No_BN",      "cls": ImprovedCNN, "params": {"use_bn": False, "use_dropout": True,  "use_residual": True,  "pooling_type": 'max'}},
        {"name": "CNN_No_Drop",    "cls": ImprovedCNN, "params": {"use_bn": True,  "use_dropout": False, "use_residual": True,  "pooling_type": 'max'}},
        {"name": "CNN_No_Res",     "cls": ImprovedCNN, "params": {"use_bn": True,  "use_dropout": True,  "use_residual": False, "pooling_type": 'max'}},
        {"name": "CNN_Stride",     "cls": ImprovedCNN, "params": {"use_bn": True,  "use_dropout": True,  "use_residual": True,  "pooling_type": 'stride'}},
        
        # --- NN Ablations ---
        {"name": "NN_Full",        "cls": ImprovedNN,  "params": {"use_bn": True,  "use_dropout": True}},
        {"name": "NN_No_BN",       "cls": ImprovedNN,  "params": {"use_bn": False, "use_dropout": True}},
        {"name": "NN_No_Drop",     "cls": ImprovedNN,  "params": {"use_bn": True,  "use_dropout": False}},
    ]
    
    results = {}
    
    # 3. Run Loop
    for exp in experiments_list:
        name = exp['name']
        cls = exp['cls']
        params = exp['params']
        results[name] = run_experiment(name, cls, params, train_loader, val_loader, epochs=25)
        
    # 4. Plot & Save
    print("\n📊 Generating Report Artifacts...")
    plot_results(results, output_dir='output')
    save_summary_table(results, output_dir='output')
    
    print("\n✨ All Ablation Studies Completed! Check 'output/' folder.")

if __name__ == "__main__":
    main()
