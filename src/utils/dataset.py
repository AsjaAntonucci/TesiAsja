"""
═══════════════════════════════════════════════════════════════════════════════
                         DATASET - Data Loading Utilities
═══════════════════════════════════════════════════════════════════════════════

This module provides functions for:
  - Loading preprocessed EEG data for each patient
  - Splitting data into train/test sets
  - Creating DataLoaders for training with PyTorch
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def load_patient_data(patient_id, data_dir="./data/preprocessed"):
    path = Path(data_dir) / f"{patient_id}_preprocessed.pt"
    d    = torch.load(path)
    return d['X'], d['y']

def get_loaders(X, y, test_size=0.2, batch_size=32):
    n               = len(y)
    split           = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
