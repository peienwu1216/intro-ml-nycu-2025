# Image Aesthetic Assessment Experiment Framework

This framework is designed to allow researchers to implement new model architectures while ensuring fair comparison with the SAMPNet baseline and other experiments.

## Directory Structure

```
test_format/
├── config.py           # Configuration (Hyperparameters, Paths)
├── train.py            # Main training loop (Entry point)
├── core/               # [Control Variables] Core logic that should NOT change
│   ├── dataset.py      # Data loading, splitting, and augmentation
│   ├── loss.py         # Loss function definitions (EMD, Rank, Attribute)
│   └── utils.py        # Metric calculations (SRCC, LCC, Accuracy)
└── models/             # [Independent Variables] Where you implement new models
    ├── base_model.py   # Interface definition
    └── convnext_v2.py  # Example implementation (Current SOTA)
```

## How to Create a New Experiment

1.  **Create a New Model File**:
    *   Create a new file in `models/`, e.g., `models/my_new_model.py`.
    *   Inherit from `BaseModel` (defined in `models/base_model.py`).
    *   Implement the `__init__` and `forward` methods.
    *   Ensure your `forward` method returns `(score_distribution, attributes)`.

2.  **Update Configuration**:
    *   Open `config.py`.
    *   Change `EXP_NAME` to a unique name for your experiment.
    *   Adjust `BACKBONE` or other hyperparameters if necessary.

3.  **Register Model in Training Loop**:
    *   Open `train.py`.
    *   Import your new model class.
    *   Update the model initialization line:
        ```python
        # model = ConvNeXtV2SAMPNet(cfg).to(device)
        model = MyNewModel(cfg).to(device)
        ```

4.  **Run Training**:
    *   Execute `python train.py`.

## Alignment with SAMPNet

To ensure fair comparison with the original SAMPNet paper, the following components are strictly controlled ("Control Variables"):

1.  **Dataset Split**:
    *   `core/dataset.py` loads the exact same `split.json` as SAMPNet.
    *   **Do not modify** the split logic.

2.  **Saliency Detection**:
    *   `core/dataset.py` uses the Spectral Residual algorithm, identical to SAMPNet's implementation.

3.  **Evaluation Metrics**:
    *   `core/utils.py` implements SRCC, LCC, and Accuracy exactly as in SAMPNet.
    *   Accuracy threshold is fixed at **2.6**.
    *   Score calculation (`dist2ave`) uses the weighted sum of probabilities $[1, 5]$.

4.  **Loss Functions**:
    *   `core/loss.py` implements the Earth Mover's Distance (EMD) loss with $r=2$, consistent with SAMPNet.

## What You Can Change (Independent Variables)

*   **Model Architecture**: Anything inside `models/`.
*   **Feature Fusion Strategy**: How you combine image features with saliency maps.
*   **Pooling Strategy**: Replacing GAP with Attention, GRN, etc.
*   **Hyperparameters**: Learning rate, batch size, optimizer (in `config.py`).
*   **Augmentation**: You can enhance augmentation in `core/dataset.py` (e.g., ColorJitter), but keep the basic Resize/Normalize consistent.

## Current Best Practice (ConvNeXt V2)

The provided example `models/convnext_v2.py` implements:
*   **Backbone**: ConvNeXt V2 Nano (Pretrained).
*   **SGFM**: Saliency-Guided Feature Modulation.
*   **GRN Pooling**: Global Response Normalization for spatial attention.
*   **EMA**: Exponential Moving Average for stable training.
