# Image Aesthetic Assessment: Beyond the Eye of the Beholder
## Final Project Report

### 1. Dataset Description
*   **Source**: CADB (Composition-aware Aesthetic Database).
*   **Overview**: A dataset designed for understanding image aesthetics through composition and high-level attributes, going beyond simple "good/bad" binary labels.
*   **Size & Type**: 
    *   **Total Images**: 9,497 (Train: 8,547 / Test: 950).
    *   **Data Type**: High-resolution RGB images paired with tabular annotation files.
*   **Data Structure (JSON Schema)**:
    *   **`composition_scores.json`**: Contains the ground truth aesthetic scores.
        ```json
        "10003.jpg": {
            "scores": [2.0, 3.0, 2.0, 2.0, 2.0],  // Raw ratings from 5 experts
            "dist": [0.0, 0.8, 0.2, 0.0, 0.0],    // Normalized probability distribution (1-5)
            "mean": 2.2                           // Mean aesthetic score
        }
        ```
    *   **`composition_attributes.json`**: Contains 12 high-level photographic attributes (normalized -1 to 1).
        ```json
        "965.jpg": {
            "Light": -0.2,            "Symmetry": 0.0,
            "Object": -0.6,           "RuleOfThirds": -0.2,
            "Repetition": 0.0,        "BalacingElements": 0.0,
            "ColorHarmony": 0.0,      "MotionBlur": -0.4,
            "VividColor": 0.2,        "DoF": -0.2,
            "Content": -0.4,          "score": 0.25
        }
        ```
*   **Features & Annotations**:
    *   **Aesthetic Scores**: Each image is rated by **5 individual expert annotators**. Instead of a single mean score, we utilize the **distribution** of these 5 scores to capture the consensus (low variance) or controversy (high variance) of an image.
    *   **Attributes**: Experts annotated **12 specific photographic attributes**, including:
        *   **Composition**: Rule of Thirds, Symmetry, Balancing Elements, Repetition.
        *   **Lighting & Color**: Light, Color Harmony, Vivid Color.
        *   **Technique**: Depth of Field (DoF), Motion Blur.
        *   **Content**: Object Emphasis.
*   **Learning Task**:
    *   **Primary**: **Label Distribution Learning (LDL)** (Regression). We predict the probability distribution of scores (1-5) rather than a single scalar.
    *   **Secondary**: **Ranking**. We aim to correctly order images by aesthetic quality (A > B).

### 2. Preprocessing & Feature Engineering
*   **Saliency Map Generation**:
    *   **Algorithm**: **Spectral Residual (SR)**.
    *   **Purpose**: To simulate the human visual system's "pre-attentive" mechanism, identifying potential regions of interest (ROI) before detailed processing.
    *   **Process**:
        1.  **Log Spectrum**: Compute the FFT of the image and take the log amplitude.
        2.  **Spectral Residual**: Subtract the averaged log spectrum (local average) from the original log spectrum to remove redundant information (background).
        3.  **Saliency Map**: Reconstruct the image using the residual spectrum and original phase via Inverse FFT.
    *   **Code Implementation**:
        ```python
        def detect_saliency(img, scale=6, q_value=0.95, target_size=(384, 384)):
            # 1. Preprocessing: Convert to Grayscale & Resize
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            W, H = img_gray.shape
            img_resize = cv2.resize(img_gray, (H // scale, W // scale))

            # 2. Spectral Residual Calculation
            myFFT = np.fft.fft2(img_resize)
            myLogAmplitude = np.log(np.abs(myFFT) + 1e-6)
            myAvg = cv2.blur(myLogAmplitude, (3, 3))
            mySpectralResidual = myLogAmplitude - myAvg

            # 3. Reconstruction (Inverse FFT)
            m = np.exp(mySpectralResidual) * (np.cos(np.angle(myFFT)) + 1j * np.sin(np.angle(myFFT)))
            saliencyMap = np.abs(np.fft.ifft2(m)) ** 2
            
            # 4. Post-processing: Gaussian Blur & Normalization
            saliencyMap = cv2.GaussianBlur(saliencyMap, (9, 9), 2.5)
            saliencyMap = cv2.resize(saliencyMap, target_size)
            
            # Thresholding to remove weak activations
            threshold = np.quantile(saliencyMap, q_value)
            saliencyMap[saliencyMap > threshold] = threshold
            return (saliencyMap - saliencyMap.min()) / threshold
        ```
*   **Image Resizing (Warping vs. Cropping)**:
    *   **Strategy**: We employ **Warping** (resizing the entire image to 384x384) rather than Random Crop.
    *   **Justification**: Aesthetic assessment relies heavily on global composition (e.g., Rule of Thirds, Balancing Elements). Random cropping destroys these compositional structures, rendering the label invalid. Warping preserves the relative positions of elements.
*   **Normalization**:
    *   Images are normalized using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`) to facilitate transfer learning from pretrained models.
*   **Label Distribution Construction**:
    *   We convert the 5 raw expert ratings into a normalized probability distribution. This allows us to use **EMD (Earth Mover's Distance)** loss, which is more robust for subjective tasks than MSE.

### 3. Model Implementation & Comparison
We implemented and compared four distinct phases of architecture evolution to solve the aesthetic assessment problem.

#### Phase 1: Baseline (SAMPNet / ResNet-50)

*   **Architecture Details**:
    *   **Backbone**: **ResNet-50** (Pretrained on ImageNet).
    *   **Input**: RGB Image (224x224) + Saliency Map (56x56).
    *   **Feature Extraction**: Extracts features from the last convolutional layer (2048 channels).
*   **Data Flow**:
    1.  **Image Branch**: `Image -> ResNet-50 -> Feature Map (2048x7x7)`.
    2.  **Saliency Branch**: `Saliency Map -> MaxPool -> Downsampled Map (56x56)`.
    3.  **Fusion**: The saliency map is flattened and concatenated with the global average pooled image features.
    4.  **Prediction**: `Concat(Features, Saliency) -> FC Layers -> Score`.
*   **Key Parameters**:
    *   **Input Channels**: 2048 (ResNet-50).
    *   **Fusion Method**: Concatenation (`torch.cat`).
    *   **Loss**: Standard MSE / Cross-Entropy.
*   **Pros**: Simple, established baseline.
*   **Cons**: Lacks mechanism to handle long-range dependencies; concatenation is a naive fusion strategy that introduces noise.

#### Phase 2: Swin Transformer (Swin-T)

*   **Architecture Details**:
    *   **Backbone**: **Swin Transformer Tiny** (Swin-T).
    *   **Input**: RGB Image (224x224).
    *   **Heads**: Multi-Task Learning (MTL) heads for Composition (C), Light (L), Focus (F), Originality (O), and Global Score (IAS).
*   **Data Flow**:
    1.  **Patch Partition**: Image split into 4x4 patches.
    2.  **Stage 1-4**: Hierarchical feature extraction with **Shifted Window Attention**.
    3.  **Feature Selection**: Extracts `feat_low` (Stage 2, 192 dim) and `feat_high` (Stage 4, 768 dim).
    4.  **MTL Heads**:
        *   `feat_high -> Head_C, Head_L`.
        *   `feat_low -> Head_F`.
        *   `Concat(feat_high, CLIP_Embedding) -> Head_O`.
    5.  **Global Prediction**: `Concat(C, L, F, O, feat_high) -> Head_IAS`.
*   **Key Parameters**:
    *   **Embed Dim**: 96.
    *   **Stage Dims**: 96 -> 192 -> 384 -> 768.
    *   **Window Size**: 7x7.
*   **Innovation**: Introduction of **Rank Loss** to explicitly optimize for the relative order of images (SRCC).
*   **Code Implementation (Rank Loss)**:
    ```python
    class RankLoss(nn.Module):
        def __init__(self, margin=0.0):
            super().__init__()
            self.margin = margin
            
        def forward(self, preds, targets):
            # Pairwise difference matrix
            diff_preds = preds.unsqueeze(1) - preds.unsqueeze(0)
            diff_targets = targets.unsqueeze(1) - targets.unsqueeze(0)
            
            # Sign of target difference (-1, 0, 1)
            target_sign = torch.sign(diff_targets)
            
            # Hinge Loss: max(0, margin - sign * (pred_i - pred_j))
            loss = torch.relu(self.margin - target_sign * diff_preds)
            return loss.mean()
    ```
*   **Pros**: Better at capturing context than CNNs; Rank Loss improves monotonicity.

#### Phase 3: Swin-T Opt (Refinement)

*   **Architecture Details**:
    *   **Backbone**: Swin-T.
    *   **Module**: **SAMP (Saliency-Aware Multi-Pattern)** Module.
    *   **Input**: RGB Image + Saliency Map.
*   **Data Flow**:
    1.  **Layout Queries**: Defines 16 learnable queries representing different composition patterns (e.g., Rule of Thirds grid, Symmetry axis).
    2.  **Pattern Matching**:
        *   **Triangular Pattern**: Extracts features from upper/lower triangles.
        *   **Cross Pattern**: Extracts features from center-crossing lines.
        *   **Surround Pattern**: Separates center vs. peripheral features.
    3.  **Saliency Weighting**: Instead of concatenation, saliency is used as a spatial weight: `Feature * (1 + Saliency)`.
*   **Key Parameters**:
    *   **Patterns**: Triangular, Cross, Surround, Horizontal, Vertical.
    *   **Fusion**: Weighted Sum of pattern features.
*   **Code Implementation (Layout Pattern)**:
    ```python
    class TriangularPattern(nn.Module):
        def forward(self, x, s):
            # Extract features from upper and lower triangles
            up_indices = torch.triu_indices(x.shape[2], x.shape[3], offset=1)
            up_feat = x[:,:,up_indices[0], up_indices[1]].mean(dim=2)

            lw_indices = torch.tril_indices(x.shape[2], x.shape[3], offset=-1)
            lw_feat = x[:,:,lw_indices[0], lw_indices[1]].mean(dim=2)
            
            # Fuse features to represent triangular composition
            fused = torch.stack([up_feat, lw_feat], dim=2).unsqueeze(3)
            return fused
    ```
*   **Pros**: Explicit modeling of composition layouts.

#### Phase 4: Final Form (ConvNeXt V2)

*   **Architecture Details**:
    *   **Backbone**: **ConvNeXt V2 Nano** (Pretrained).
    *   **Input**: RGB Image (384x384) + Saliency Map (384x384).
    *   **Modules**: SGFM + GRN-Aware Pooling.
*   **Data Flow**:
    1.  **Feature Extraction**: ConvNeXt V2 extracts multi-scale features (Stage 3: 320 dim, Stage 4: 640 dim).
    2.  **Adaptation**: 1x1 Conv projects features to **512 dim**.
    3.  **SGFM (Saliency-Guided Feature Modulation)**:
        *   Saliency map is downsampled to match feature size.
        *   Affine parameters ($\gamma, \beta$) are learned from saliency.
        *   Modulation: $F_{out} = F_{in} \cdot (1 + \gamma) + \beta$.
    4.  **GRN-Aware Pooling**:
        *   **Spatial Mixing**: Depthwise Conv.
        *   **GRN**: Global Response Normalization to boost feature competition.
        *   **Attention**: Spatial Softmax to pool features into a global vector.
    5.  **Prediction**: FC Layers -> 5-class Probability Distribution.
*   **Key Parameters**:
    *   **Backbone**: `convnextv2_nano`.
    *   **Target Dim**: 512.
    *   **GRN Epsilon**: 1e-6.
    *   **Dropout**: 0.1 - 0.5 (tuned).
*   **Code Implementation (SGFM & GRN)**:
    ```python
    class SGFM(nn.Module):
        """ Saliency-Guided Feature Modulation """
        def __init__(self, dim):
            super().__init__()
            self.conv_gamma = nn.Conv2d(1, dim, kernel_size=3, padding=1)
            self.conv_beta = nn.Conv2d(1, dim, kernel_size=3, padding=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, saliency):
            # Resize saliency to match feature map size
            saliency = F.interpolate(saliency, size=x.shape[2:], mode='bilinear')
            
            # Learn affine parameters from saliency
            gamma = self.sigmoid(self.conv_gamma(saliency))
            beta = self.conv_beta(saliency)
            
            # Modulate features: x * (1 + gamma) + beta
            return x * (1 + gamma) + beta

    class GRN(nn.Module):
        """ Global Response Normalization """
        def forward(self, x):
            # Compute L2 norm across spatial dimensions
            Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            # Normalize relative to global mean
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            # Calibrate features
            return self.gamma * (x * Nx) + self.beta + x
    ```
*   **Loss Function**:
    *   **EMD Loss**: For learning the score distribution (shape of opinion).
    *   **Rank Loss**: For optimizing ranking accuracy (relative order).
    *   **Attribute Loss**: Auxiliary task to learn compositional rules.

### 4. Results & Discussion

#### Performance Comparison
We evaluated the models using **SRCC (Spearman Rank Correlation Coefficient)** as the primary metric, as ranking capability is more important than absolute score prediction in subjective tasks.

| Model | SRCC | Improvement |
| :--- | :---: | :--- |
| **ResNet-50 (Baseline)** | 0.642 | - |
| **Swin-T** | 0.671 | +4.5% |
| **Swin-T Opt** | 0.692 | +7.8% |
| **ConvNeXt V2 (Final)** | **0.715** | **+11.3%** |

#### Ablation Studies & Micro-Design
To ensure optimality, we conducted rigorous component testing:
1.  **Fusion Strategy**: **Affine (SGFM)** outperformed Concatenation and Addition. It allows the saliency map to dynamically "scale" and "shift" features, effectively highlighting the subject without introducing additive noise.
2.  **Normalization**: **GRN (Global Response Normalization)** proved critical. In aesthetic tasks, many feature channels can become inactive. GRN forces competition, keeping essential features alive.
3.  **Activation Function**: **GELU** outperformed ReLU, providing smoother gradients for regression tasks.
4.  **Preprocessing**: **Warping** (384x384) significantly outperformed Random Crop, confirming that preserving global composition is vital for aesthetic assessment.

#### Qualitative Analysis (GradCAM)
*   **ResNet-50**: Attention maps were scattered and often focused on irrelevant background textures.
*   **ConvNeXt V2**: Attention maps tightly focused on the **Main Subject** and **Key Compositional Lines** (e.g., horizon, leading lines), aligning closely with human aesthetic perception.

#### Conclusion
The transition from CNNs to Transformers and finally to a modernized CNN (ConvNeXt V2) with specific aesthetic modules (SGFM, GRN) demonstrates that generic models are insufficient for subjective tasks. The combination of **Label Distribution Learning**, **Rank Loss**, and **Composition-Preserving Preprocessing** was key to achieving state-of-the-art performance on the CADB dataset. Our final model not only predicts scores accurately but also "looks" at the image in a human-like way.

### 5. Deliverables
*   **Written Report**: This document.
*   **Final Presentation**: See `slide.md` for the presentation structure and `inference.py` for the demo code.
