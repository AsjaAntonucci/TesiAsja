# SPDNet Seizure Detection

Epileptic seizure detection comparing **SPDNet** (Riemannian), **EpiDeNet** (CNN), and **SVM** on CHB-MIT EEG data.

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train single model
python src/training/train_spdnet.py -p chb01

# Or train all patients
./run_all.ps1
```

## 📊 Models

| Model | Input | Key Config |
|-------|-------|------------|
| **SPDNet** | 21×21 SPD matrices | BiMap , 500 epochs, batch=32 |
| **EpiDeNet** | 21×256 raw signal | 5 CNN blocks, 150 epochs, batch=64 |
| **SVM** | 504 spectral features | RBF kernel, 5-fold CV |

All use 5-fold cross-validation on 21-channel EEG (256 Hz, 1-second windows).

## 📁 Data

**CHB-MIT Dataset**: 21 bipolar EEG channels, ~900 samples per patient
- Raw: `Dataset_CHB-MIT/`
- Preprocessed: `data/preprocessed/`
- Results: `results/` (checkpoints, metrics, plots)

## ⚙️ Config

Adjust hyperparameters in `src/models/{model}.py`:
- `MODEL_PARAMS`: Architecture settings
- `TRAIN_PARAMS`: Epochs, learning rate, batch size, etc.

---


