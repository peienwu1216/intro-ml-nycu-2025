# Image Aesthetic Assessment Project

This repository contains the code for the "Image Aesthetic Assessment: Beyond the Eye of the Beholder" project. It includes the implementation of four evolutionary phases of the model and a web application demo.

## Directory Structure

*   **`phase1_baseline/`**:
    *   Contains the code for Phase 1 (Baseline), based on ResNet-50 and SAMPNet.
    *   Key files: `samp_net.py`, `train.py`, `cadb_dataset.py`.
*   **`phase2_swin/`**:
    *   Contains the code for Phase 2 (Swin Transformer).
    *   Key files: `model.py` (Swin-T + RankLoss), `train.py`.
*   **`phase3_swin_opt/`**:
    *   Contains the code for Phase 3 (Swin-T Optimization).
    *   Key files: `model.py` (Swin-T + Layout Pattern), `train.py`.
*   **`phase4_convnext/`**:
    *   Contains the code for Phase 4 (Final Form), based on ConvNeXt V2.
    *   Key files: `model.py` (ConvNeXt + SGFM + GRN), `train.py`.
*   **`app_demo/`**:
    *   Contains the Flask web application and frontend code.
    *   Key files: `app.py`, `frontend/`.

## How to Retrain

### Prerequisites
Ensure you have the required Python packages installed:
```bash
pip install -r requirements.txt
```
(Note: You may need to install specific versions of `torch`, `torchvision`, `timm`, etc., depending on your CUDA version.)

### Dataset Preparation
Ensure the CADB dataset is located in the correct path as expected by the `dataset.py` or `cadb_dataset.py` in each phase folder. You might need to adjust the `root` path in the configuration or arguments.

### Phase 1: Baseline
```bash
cd phase1_baseline
python train.py --epoch 50 --batch_size 32 --lr 1e-4
```

### Phase 2: Swin Transformer
```bash
cd phase2_swin
python train.py --epochs 50 --batch_size 32 --lr 1e-4
```

### Phase 3: Swin-T Opt
```bash
cd phase3_swin_opt
python train.py --epochs 50 --batch_size 32 --lr 1e-4
```

### Phase 4: ConvNeXt V2 (Final)
```bash
cd phase4_convnext
python train.py --epochs 50 --batch_size 32 --lr 1e-4
```

## How to Run the App

1.  **Backend Setup**:
    Navigate to the `app_demo` directory and install Python dependencies:
    ```bash
    cd app_demo
    pip install -r requirements.txt
    ```

2.  **Frontend Setup**:
    Navigate to the `frontend` directory, install Node.js dependencies, and build the static files:
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

3.  **Model Preparation**:
    Ensure the model weights (`best_model.pth`) are placed in the `model/` directory.

4.  **Run the Application**:
    Start the Flask server:
    ```bash
    python app.py
    ```

5.  **Access**:
    Open your browser and go to `http://localhost:5000` (or the port specified in the console).

## Notes
*   The training scripts assume a specific dataset structure. Please refer to the `dataset.py` in each folder for details.
*   Hyperparameters (learning rate, batch size, etc.) can be adjusted in `config.py` or via command-line arguments, depending on the implementation in each phase.
