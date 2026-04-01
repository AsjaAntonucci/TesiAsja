"""
═══════════════════════════════════════════════════════════════════════════════
                         DATASET - Data Loading Utilities
═══════════════════════════════════════════════════════════════════════════════

Questo modulo fornisce funzioni per:
  - Caricare i dati EEG preprocessati di ciascun paziente
  - Dividere i dati in train/test
  - Creare DataLoader per il training con PyTorch
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def load_patient_data(patient_id, data_dir="./data/preprocessed"):
    # Costruisce il percorso al file del paziente
    path = Path(data_dir) / f"{patient_id}_preprocessed.pt"
    # Carica il dizionario salvato
    d    = torch.load(path)
    # Ritorna i segnali (X) e le etichette (y)
    return d['X'], d['y']   # (N, 21, 256), (N,)

def get_loaders(X, y, test_size=0.2, batch_size=32):
    # Numero totale di campioni
    n               = len(y)
    # Calcola l'indice di split (es. 80% train, 20% test)
    split           = int(n * (1 - test_size))
    # Divide X e y secondo lo split
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Crea il DataLoader per il training (shuffle=True mescola i dati ad ogni epoca)
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    # Crea il DataLoader per il test (shuffle=False mantiene l'ordine)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
