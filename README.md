# SPDNet Seizure Detection

This project implements an epileptic seizure detection system based on **SPDNet (Symmetric Positive Definite Network)** using the **CHB-MIT** dataset.

## 📋 Description

This project trains a deep learning model on EEG data to classify signal segments as **seizure** or **non-seizure** (normal). It uses symmetric positive definite (SPD) matrices as features, which capture the covariance structure of EEG signals.

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   ├── spdnet_classic.py       # SPDNet (Symmetric Positive Definite Network)
│   │   ├── cnn_epidenet.py         # EpiDeNet (Convolutional Neural Network)
│   │   └── ml_baseline.py          # ML Baseline (SVM with spectral features)
│   └── training/
│       ├── train_spdnet.py         # Training script for SPDNet
│       ├── train_cnn.py            # Training script for EpiDeNet
│       └── train_ml.py             # Training script for ML Baseline
├── utils/
│   ├── dataset.py                  # Dataset loading and management
│   ├── preprocessing.py            # EEG signal preprocessing
│   └── metrics.py                  # Evaluation metrics
├── data/
│   └── preprocessed/               # Preprocessed data in .pt format
├── Dataset_CHB-MIT/                # Raw EEG dataset (files .edf)
├── results/
│   ├── checkpoints/                # Saved models
│   ├── metrics/                    # Results CSV
│   ├── params/                     # Model parameters JSON
│   └── plots/                      # Training curves and cross-validation plots
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

1. **Clone the repository** (if applicable) or navigate to the project folder

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

This project implements **three different neural network approaches** for seizure detection, allowing comparison between different architectures:

### 1️⃣ SPDNet (Symmetric Positive Definite Network)

Train the main SPDNet model on a single patient:

```bash
python src/training/train_spdnet.py -p chb01
```

**Architecture Overview:**
```
CovLayer → Shrinkage → BiMap → ReEig → BiMap → ReEig → LogEig → Linear
```

**Technical Parameters:**
- **Input**: 21 EEG channels → 21×21 SPD covariance matrices
- **Output**: 2 classes (seizure / non-seizure)
- **BiMap Dimensions**: 21 → 10 → 5 → 2
- **Tangent Space Dimension**: 3 (from final 2×2 SPD matrix)
- **Architecture Depth**: 3 BiMap + ReEig blocks
- **Reference**: Huang & Van Gool (AAAI 2017)
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Loss**: CrossEntropyLoss
- **Epochs**: 500 (with early stopping)
- **Early Stopping Patience**: 30 epochs
- **Batch Size**: 32
- **Train/Test Split**: 80%/20% (temporal)
- **Cross-Validation**: 5-fold StratifiedKFold

**Output Files:**
- `results/checkpoints/{PATIENT_ID}_spdnet_classic.pt` — Trained model
- `results/params/{PATIENT_ID}_spdnet_params.json` — Model parameters
- `results/metrics/{PATIENT_ID}_spdnet_results.csv` — Performance metrics
- `results/plots/{PATIENT_ID}_spdnet_losses.png` — Training curves

---

### 2️⃣ EpiDeNet (Convolutional Neural Network)

Train the EpiDeNet (CNN) model on a single patient:

```bash
python src/training/train_cnn.py -p chb01
```

**Architecture Overview:**
```
[5× Conv2D + BatchNorm + ReLU + MaxPool] → GlobalAvgPool → Linear
```

**Technical Parameters:**
- **Input Shape**: (batch, 21, 256)
- **Convolutional Stack** (5 blocks):
  - **Block 1-3 (Temporal Filters)**: Kernel (1,k) — captures short-term temporal patterns
    - Block 1: 1→8 filters, kernel (1,4), pool (1,4)
    - Block 2: 8→16 filters, kernel (1,8), pool (1,2)
    - Block 3: 16→32 filters, kernel (1,4), pool (1,2)
  - **Block 4 (Spatial Filter)**: Kernel (3,1) — aggregates information across 21 channels
    - Block 4: 32→32 filters, kernel (3,1), pool (21,1)
  - **Block 5 (Compression)**: Final feature refinement
    - Block 5: 32→64 filters, kernel (1,1), pool (1,1)
- **Global Average Pooling**: Collapses to 64-dim feature vector
- **Classifier**: Linear(64, 2)
- **Temporal Reduction**: 256 → 64 → 32 → 16 (then spatial collapse)
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Loss**: CrossEntropyLoss
- **Epochs**: 150 (with early stopping)
- **Early Stopping Patience**: 10 epochs
- **Batch Size**: 64
- **Train/Test Split**: 80%/20% (temporal)
- **Validation Split (final training)**: 90%/10%
- **Cross-Validation**: 5-fold StratifiedKFold
- **Reference**: Adapted from Ferrara PhD Thesis (2025) — 1D separable kernels optimized for 1-second windows

**Output Files:**
- `results/checkpoints/{PATIENT_ID}_epidenet_cnn.pt` — Trained model
- `results/params/{PATIENT_ID}_epidenet_params.json` — Model parameters
- `results/metrics/{PATIENT_ID}_epidenet_results.csv` — Performance metrics
- `results/plots/{PATIENT_ID}_epidenet_losses.png` — Training curves

---

### 3️⃣ ML Baseline (Classical Machine Learning with SVM)

Train the ML baseline model (SVM with spectral features) on a single patient:

```bash
python src/training/train_ml.py -p chb01
```

**Feature Engineering:**
```
EEG Signal → [8 frequency bands × 21 channels × W temporal stack] → SVM Classification
```

**Frequency Bands (8 bands from 0.5–25 Hz):**
The frequency space is divided uniformly into 8 bands using `np.linspace(0.5, 25, 9)`:
- **Band 1**: 0.5 – 3.5 Hz    (sub-delta)
- **Band 2**: 3.5 – 6.5 Hz    (delta/theta)
- **Band 3**: 6.5 – 9.5 Hz    (theta/alpha)
- **Band 4**: 9.5 – 12.5 Hz   (alpha)
- **Band 5**: 12.5 – 15.5 Hz  (alpha/beta)
- **Band 6**: 15.5 – 18.5 Hz  (beta)
- **Band 7**: 18.5 – 21.5 Hz  (beta)
- **Band 8**: 21.5 – 25 Hz    (beta/gamma)

**Temporal Stacking (W):**
Each sample's spectral features are stacked across W=3 consecutive windows for temporal context:
- Feature matrix per window: 21 channels × 8 bands = 168 features
- After temporal stacking (W=3): 168 × 3 = **504 features total**

**Technical Parameters:**
- **Feature Extraction**: Power spectral density (Welch method)
- **Total Features**: 504 (21 channels × 8 bands × 3 temporal windows)
- **Sampling Rate**: 256 Hz
- **FFT Window**: 256 samples (~1 second)
- **Temporal Stack (W)**: 3 consecutive windows
- **SVM Configuration**:
  - Kernel: RBF
  - C: 1.0
  - Gamma: scale
  - Class weight: balanced
- **Cross-Validation**: 5-fold StratifiedKFold
- **Train/Test Split**: 80%/20%
- **Scaler**: StandardScaler (per-fold)

**Output Files:**
- `results/checkpoints/{PATIENT_ID}_ml_baseline.pkl` — Trained SVM model
- `results/params/{PATIENT_ID}_ml_params.json` — Feature and SVM parameters
- `results/metrics/{PATIENT_ID}_ml_results.csv` — Performance metrics (CV scores)
- `results/plots/{PATIENT_ID}_ml_cv_scores.png` — Cross-validation curves

---

### Batch Training (All Patients)

To train all three models across all patients, use:

```bash
./run_all.ps1
```

This PowerShell script trains SPDNet, EpiDeNet, and ML baseline models on all patients in the dataset.

## 📊 Model Details & Comparison

| Aspect | SPDNet | EpiDeNet (CNN) | ML Baseline |
|--------|--------|---|---|
| **Type** | Deep Learning (SPD Networks) | Deep Learning (CNN) | Classical ML (SVM) |
| **Input** | 21×21 SPD covariance matrices | 21×256 raw signal tensor | 21 channels × 8 freq bands × 3 windows |
| **Feature Extraction** | Implicit (learned via BiMap & ReEig) | Implicit (learned via conv filters) | Explicit (Power spectral density + stacking) |
| **Hidden/Subspace Dims** | BiMap: 21→10→5→2, Tangent(3) | Conv: 1→8→16→32→32→64 | 504 spectral features |
| **Kernel/Layers** | SPD BilinearMap + ReEig + LogEig | Conv2D 1D-separable (temporal+spatial) | RBF SVM |
| **Epochs** | 500 (early stop @30) | 150 (early stop @10) | 5-fold CV |
| **Batch Size** | 32 | 64 | Full batch per fold |
| **Computational Cost** | Medium | Low–Medium | Very Low |
| **Interpretability** | Low (black-box) | Low (black-box) | High (spectral features) |
| **Reference** | Huang & Van Gool (AAAI 2017) | Ferrara PhD (2025) + Sensors 2025 | Classical ML baseline |

---

### SPDNet (Main Model)

- **Architecture**: SPDNet (Symmetric Positive Definite Network)
- **Input**: 21 EEG channels → 21×21 SPD covariance matrices
- **BiMap Reduction Path**: 21 → 10 → 5 → 2 (3 stages)
- **Tangent Space Dimension**: 3 (from final 2×2 SPD matrix)
- **Key Layers**: Shrinkage, BilinearMap (×3), ReEig (×3), LogEig
- **Output**: 2 classes (seizure / non-seizure)
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Loss**: CrossEntropyLoss
- **Epochs**: 500 (with early stopping patience=30)
- **Batch Size**: 32
- **Cross-Validation**: 5-fold StratifiedKFold
- **Split**: 80% train, 20% test

### EpiDeNet (CNN Variant)

- **Architecture**: 5-block convolutional network with 1D-separable kernels
- **Input**: 21 channels × 256 timepoints
- **Temporal Filters (Blocks 1-3)**: Kernel (1,k) — captures short-term patterns
  - Block 1: 1→8 filters, kernel (1,4), pool (1,4)
  - Block 2: 8→16 filters, kernel (1,8), pool (1,2)
  - Block 3: 16→32 filters, kernel (1,4), pool (1,2)
- **Spatial Filter (Block 4)**: Kernel (3,1) — aggregates across channels
  - Block 4: 32→32 filters, kernel (3,1), pool (21,1)
- **Compression (Block 5)**: Final refinement
  - Block 5: 32→64 filters, kernel (1,1)
- **Global Average Pooling**: Reduces to 64-dim vector
- **Classifier**: Linear(64, 2)
- **Temporal Reduction Pattern**: 256 → 64 → 32 → 16 samples
- **Output**: 2 classes (seizure / non-seizure)
- **Optimizer**: Adam (lr=1e-3, weight_decay=0.0)
- **Loss**: CrossEntropyLoss
- **Epochs**: 150 (with early stopping patience=10)
- **Batch Size**: 64
- **Validation Split (final)**: 90% train / 10% val
- **Cross-Validation**: 5-fold StratifiedKFold

### ML Baseline (SVM + Spectral Features)

- **Feature Space**: Power spectral density across 8 frequency bands per channel with temporal stacking
  - 8 uniform frequency bands: 0.5–25 Hz
  - 21 channels × 8 bands = 168 features per window
  - Temporal stacking (W=3): 168 × 3 = **504 total features**
- **Welch Parameters**: nperseg=256 (1 second), fs=256 Hz
- **Classifier**: Support Vector Machine with RBF kernel
  - Kernel: RBF
  - C: 1.0
  - Gamma: scale
  - Class weight: balanced
- **Training**: Standard split (80/20) + 5-fold StratifiedKFold cross-validation
- **Scaling**: StandardScaler applied per fold
- **Advantage**: Transparent, interpretable, fast
- **Baseline**: For comparison with neural network approaches

## 🔬 Data Preprocessing Pipeline

1. **Load EEG File** (.edf)
   - Read 21-channel recordings
   - Native sampling rate: 256 Hz
   
2. **Extract Covariance Structure**
   - Compute empirical covariance matrix per segment → 21×21 SPD matrices
   - Applied to all three models for consistency
   
3. **Normalization & Preprocessing**
   - Standardization per channel
   - Padding/truncation to fixed length (256 timepoints)
   
4. **Save as PyTorch Tensors**
   - Format: `.pt` files with X (features) and y (labels)
   - Storage location: `data/preprocessed/{PATIENT_ID}_preprocessed.pt`

---

## 📝 Parameter Management (Updated March 2026)

Each model exposes all parameters explicitly for reproducibility:

### SPDNet Parameters
```python
MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,
    'init_shrinkage': 0.1,
    'bimap1_out': 10,           # 21 → 10
    'bimap2_out': 5,            # 10 → 5
    'bimap3_out': 2,            # 5 → 2
    'tangent_dim': 3,           # from 2×2 SPD matrix
    'architecture': 'CovLayer → Shrinkage → BiMap×3 → ReEig×3 → LogEig → Linear',
    'reference': 'Huang & Van Gool (AAAI 2017)'
}

TRAIN_PARAMS = {
    'optimizer': 'Adam',
    'lr': 1e-3,
    'weight_decay': 0.0,
    'loss': 'CrossEntropyLoss',
    'epochs': 500,              # increased from 50
    'patience': 30,             # early stopping
    'batch_size': 32,
    'train_split': 0.8,
    'cv_folds': 5,
    'cv_strategy': 'StratifiedKFold',
    'scoring': 'balanced_accuracy',
    'seed': 42
}
```

### EpiDeNet Parameters
```python
MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,
    'input_shape': '(batch, 1, 21, 256)',
    'blocks': [
        {'in_ch': 1,  'out_ch': 8,  'conv_kernel': [1, 4], 'pool_kernel': [1, 4], 'role': 'temporal'},
        {'in_ch': 8,  'out_ch': 16, 'conv_kernel': [1, 8], 'pool_kernel': [1, 2], 'role': 'temporal'},
        {'in_ch': 16, 'out_ch': 32, 'conv_kernel': [1, 4], 'pool_kernel': [1, 2], 'role': 'temporal'},
        {'in_ch': 32, 'out_ch': 32, 'conv_kernel': [3, 1], 'pool_kernel': [21, 1], 'role': 'spatial'},
        {'in_ch': 32, 'out_ch': 64, 'conv_kernel': [1, 1], 'pool_kernel': [1, 1], 'role': 'compression'},
    ],
    'classifier_in': 64,
    'reference': 'Ferrara PhD (2025) + Sensors 2025 — 1D separable kernels'
}

TRAIN_PARAMS = {
    'optimizer': 'Adam',
    'lr': 1e-3,
    'weight_decay': 0.0,
    'loss': 'CrossEntropyLoss',
    'epochs': 150,              # increased from 50
    'batch_size': 64,           # increased from 32
    'train_split': 0.8,
    'val_split_final': 0.9,     # 90% train / 10% val
    'cv_folds': 5,
    'cv_strategy': 'StratifiedKFold',
    'early_stopping_patience': 10,
    'seed': 42
}
```

### ML Baseline Parameters
```python
FEATURE_PARAMS = {
    'n_channels': 21,
    'n_bands': 8,               # 8 uniform bands (0.5–25 Hz)
    'n_features': 504,          # 21 × 8 × 3 (W=3 temporal stacking)
    'fs': 256,
    'nperseg': 256,
    'W': 3,                     # temporal stacking window
    'bands': {
        'band1': (0.5, 3.5),   'band2': (3.5, 6.5),
        'band3': (6.5, 9.5),   'band4': (9.5, 12.5),
        'band5': (12.5, 15.5), 'band6': (15.5, 18.5),
        'band7': (18.5, 21.5), 'band8': (21.5, 25.0),
    }
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',           # changed from 0.1
    'class_weight': 'balanced',
    'probability': False,
}

TRAIN_PARAMS = {
    'train_split': 0.8,
    'cv_folds': 5,
    'cv_strategy': 'StratifiedKFold',
    'scoring': 'balanced_accuracy',
    'scaler': 'StandardScaler',
}
```

All parameters are saved to `results/params/{PATIENT_ID}_{model}_params.json` for full traceability.

## 📈 Evaluation Metrics

- **Balanced Accuracy**: Average of sensitivity and specificity
  - Formula: $(TPR + TNR) / 2$
  - Handles class imbalance well
  
- **Sensitivity (True Positive Rate / Recall)**: Ability to detect seizures
  - Formula: $TP / (TP + FN)$
  - Critical for medical applications
  
- **Specificity (True Negative Rate)**: Ability to correctly identify non-seizures
  - Formula: $TN / (TN + FP)$
  
- **Confusion Matrix**: TP, FP, TN, FN
- **Cross-Validation Scores** (ML Baseline): Per-fold balanced accuracy

---

## 📊 Output & Results

### Files Generated per Training Run

**Checkpoints:**
```
results/checkpoints/
├── chb01_spdnet_classic.pt          # SPDNet weights
├── chb01_epidenet_cnn.pt           # EpiDeNet weights
└── chb01_ml_baseline.pkl           # Scikit-learn SVM serialized
```

**Metrics:**
```
results/metrics/
├── chb01_spdnet_results.csv        # BA, Sensitivity, Specificity, CM
├── chb01_epidenet_results.csv      # BA, Sensitivity, Specificity, CM
└── chb01_ml_results.csv            # BA, CV scores, CM
```

**Parameters:**
```
results/params/
├── chb01_spdnet_params.json        # Model & training config
├── chb01_epidenet_params.json      # Model & training config
└── chb01_ml_params.json            # Feature & SVM config
```

**Visualizations:**
```
results/plots/
├── chb01_spdnet_losses.png         # Training & validation loss curves
├── chb01_epidenet_losses.png       # Training & validation loss curves
└── chb01_ml_cv_scores.png          # Cross-validation score distribution
```

### Example CSV Output Format

**SPDNet / EpiDeNet Results:**
```
patient_id,model,epochs_trained,final_train_loss,final_val_loss,
balanced_accuracy,sensitivity,specificity,
TP,FP,TN,FN,timestamp
chb01,spdnet_classic,50,0.234,0.456,0.89,0.92,0.86,112,18,156,14,2026-03-23T10:45:00
```

**ML Baseline (with CV):**
```
patient_id,model,cv_fold_1,cv_fold_2,cv_fold_3,cv_fold_4,cv_fold_5,
mean_cv_score,std_cv_score,balanced_accuracy,sensitivity,
TP,FP,TN,FN,timestamp
chb01,ml_baseline,0.85,0.88,0.86,0.87,0.84,0.860,0.014,0.86,0.91,125,9,145,31,2026-03-23T10:45:00
```

## � Reproducibility & Random Seeds

All randomness is controlled for reproducibility:

```python
# Fixed seeds in all training scripts
SEED = 42
random.seed(SEED)                  # Python
np.random.seed(SEED)               # NumPy
torch.manual_seed(SEED)            # PyTorch CPU
# torch.cuda.manual_seed(SEED)     # PyTorch GPU (if available)
```

**Consequences:**
- Same data → Same train/test split
- Same initialization → Identical model weights at start
- Same hyperparameters → Fully reproducible results
- Run the same command twice → Get identical metrics

---

## 🔧 Requirements

- **Python**: 3.8+
- **Deep Learning**:
  - PyTorch 2.0+
  - NumPy
  - scikit-learn
- **EEG Processing**:
  - MNE (for reading .edf files)
  - spd-learn (for SPDNet matrices)
- **Visualization**:
  - Matplotlib
  - Pandas (for CSV handling)
- **Model Serialization**:
  - joblib (for SVM pickle)

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Recommended setup for GPU acceleration (optional):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## � Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Train Single Model
```bash
# SPDNet on patient chb01
python src/training/train_spdnet.py -p chb01

# EpiDeNet on patient chb02
python src/training/train_cnn.py -p chb02

# ML Baseline on patient chb03
python src/training/train_ml.py -p chb03
```

### 3. Check Results
```bash
# View metrics
cat results/metrics/chb01_spdnet_results.csv

# View plots
# (Open plots in results/plots/ folder)
```

### 4. Batch Training (All Patients)
```bash
./run_all.ps1
```

---

## 📚 Project Context

- **Task**: Compare three seizure detection architectures on the CHB-MIT database
- **Dataset**: CHB-MIT Scalp EEG Database (publicly available on PhysioNet)
- **Patients**: CHB01–CHB24 (excluding CHB12, which has incomplete recordings)
- **Preprocessing**: Done offline and stored in `data/preprocessed/`
- **Goal**: Benchmark deep learning (SPDNet, CNN) vs. classical ML (SVM) approaches

---

## ⚙️ Advanced Configuration (Updated March 2026)

### Modify Training Parameters

Edit the MODEL_PARAMS and TRAIN_PARAMS dictionaries in each model file:

**For SPDNet:**
```python
# src/models/spdnet_classic.py
MODEL_PARAMS['bimap1_out'] = 15   # Change BiMap1 output (21 → 15)
MODEL_PARAMS['bimap2_out'] = 8    # Change BiMap2 output (15 → 8)
MODEL_PARAMS['bimap3_out'] = 3    # Change BiMap3 output (8 → 3)
TRAIN_PARAMS['epochs'] = 600      # Train longer
TRAIN_PARAMS['patience'] = 50     # Increase early stopping patience
TRAIN_PARAMS['lr'] = 5e-4         # Lower learning rate
TRAIN_PARAMS['batch_size'] = 16   # Smaller batches for more frequent updates
```

**For EpiDeNet:**
```python
# src/models/cnn_epidenet.py
# Modify the 'blocks' list to change filter counts or kernel sizes
MODEL_PARAMS['blocks'][0]['out_ch'] = 16  # Increase first layer filters (1→16)

TRAIN_PARAMS['epochs'] = 200      # Train longer
TRAIN_PARAMS['batch_size'] = 32   # Reduce from 64
TRAIN_PARAMS['early_stopping_patience'] = 20  # Allow more patience
TRAIN_PARAMS['lr'] = 5e-4         # Lower learning rate
```

**For ML Baseline:**
```python
# src/models/ml_baseline.py
FEATURE_PARAMS['W'] = 5           # Increase temporal stacking (more features)
FEATURE_PARAMS['n_bands'] = 10    # Add more frequency bands (extend BANDS dict)

SVM_PARAMS['C'] = 0.5             # Lower regularization (more flexible)
SVM_PARAMS['gamma'] = 'auto'      # Change gamma scaling
```

---

## 🐛 Troubleshooting

**Error: "Module not found"**
- Ensure you're running from the project root directory
- Check that `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` is in training scripts

**Error: "Data file not found"**
- Run preprocessing first or verify `data/preprocessed/{PATIENT_ID}_preprocessed.pt` exists
- Check patient ID format (should be `chbXX` with zero-padding)

**Error: CUDA out of memory / Out of memory**
- Reduce `batch_size` in TRAIN_PARAMS (e.g., SPDNet 32 → 16, EpiDeNet 64 → 32)
- Reduce `epochs` to decrease memory accumulation
- Use CPU-only mode (default is CPU, recommended)

**Error: "Early stopping triggered too early"**
- Increase `patience` in TRAIN_PARAMS
- Lower `lr` for more stable convergence

**Warning: Class imbalance in seizures**
- The SVM already uses `class_weight='balanced'` automatically
- For neural networks, consider adjusting loss weighting or sampling strategy

---

## 📋 Changelog & Recent Updates (March 2026)

### SPDNet Architecture
- **Expanded BiMap pipeline**: Now uses 3 sequential BiMap layers (21→10→5→2) instead of 2
- **Increased training duration**: Epochs increased from 50 to 500
- **Early stopping**: Added patience=30 to prevent overfitting
- **Tangent dimension**: Recalculated to 3 (from final 2×2 SPD matrix)

### EpiDeNet (CNN)
- **Refined kernel strategy**: Explicit 1D-separable kernels
  - Blocks 1-3: Temporal filters (1,k) focusing on time-domain patterns
  - Block 4: Spatial filter (3,1) aggregating across 21 channels
  - Block 5: Compression refinement (1,1)
- **Increased batch size**: 32 → 64 for better gradient estimates
- **Extended training**: Epochs increased from 50 to 150
- **Early stopping patience**: 10 epochs (more aggressive than SPDNet)
- **Validation split**: Added explicit val_split=0.9 for final training

### ML Baseline (SVM)
- **Enhanced feature extraction**: Expanded from 5 bands to 8 uniform bands (0.5–25 Hz)
- **Temporal stacking (W=3)**: Introduced feature stacking across 3 consecutive windows
- **Feature dimensions**: Increased from 105 to 504 features (21 × 8 × 3)
- **SVM hyperparameters**: 
  - Gamma changed from 0.1 → 'scale' (adaptive scaling)
  - Added class_weight='balanced' for handling seizure imbalance
- **Improved robustness**: Per-fold StandardScaler applied in cross-validation

### Backend & Infrastructure
- **All training scripts**: Updated to support new model configurations
- **Parameter tracking**: Complete MODEL_PARAMS and TRAIN_PARAMS exposed for reproducibility
- **Result generation**: CSV metrics now include model parameters and timestamp
- **Random seeds**: SEED=42 fixed in all scripts for reproducibility

---

## 👤 Author

Thesis Project — University of [Institution Name]

---

## 📚 References

- **Dataset**: CHB-MIT Scalp EEG Database
  - [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
  - Shoeb, A. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment
  
- **SPDNet**: 
  - Huang, Z., & Van Gool, L. (2017). "A Riemannian Network for SPD Matrix Learning"
  - AAAI Conference on Artificial Intelligence
  
- **EpiDeNet** (CNN Reference):
  - Ferrara, PhD Thesis (2025), Chapter 4.1.3.2
  - 5-layer convolutional architecture adapted for 21-channel EEG
  
- **Spectral Features**:
  - Welch's method for power spectral density estimation
  - Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*
  
- **Evaluation Metrics**:
  - Balanced Accuracy in imbalanced classification
  - Sensitivity/Specificity tradeoffs in medical diagnostics

---

## 📜 License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## 📞 Contact

For questions or issues, please refer to the thesis documentation or contact the author.
