"""
═══════════════════════════════════════════════════════════════════════════════
                    MODEL - SPDNet Classic on Riemannian Manifold
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
  Symmetric Positive Definite (SPD) covariance matrices on Riemannian manifold
  - Subspace projection: 21 channels → 8 dimensions
  - Covariance computation: 8×8 SPD matrices
  - Riemannian geometry: Geodesic distance in SPD manifold
  - Classification: Linear classifier in tangent space

DESIGN:
  Input:  (batch, 21, 256)  - 21 bipolar channels, 256 temporal samples (1 second @ 256Hz)
  Output: (batch, 2)        - Binary classification (seizure/non-seizure)
  
  Processing:
    1. Raw EEG → Subspace projection (21 → 8) → Covariance matrix (8×8)
    2. SPD manifold geometry: Symmetric, positive definite
    3. Riemannian metric: Log-Euclidean or Affine-Invariant distance
    4. Linear classification in tangent space at identity

REFERENCE:
  Based on spd_learn library - Riemannian geometry for EEG classification
  Suitable for manifold-based machine learning on SPD matrices
═══════════════════════════════════════════════════════════════════════════════
"""

import torch.nn as nn
from spd_learn.models import SPDNet


MODEL_PARAMS = {
    'n_channels':  21,
    'n_classes':   2,
    'subspacedim': 8,
}

TRAIN_PARAMS = {
    'optimizer':       'Adam',
    'lr':              1e-3,
    'weight_decay':    0.0,
    'loss':            'CrossEntropyLoss',
    'epochs':          500,
    'patience':        30,
    'batch_size':      32,
    'train_split':     0.8,
    'val_split_final': 0.9,
    'cv_folds':        5,
    'cv_strategy':     'StratifiedKFold',
    'scoring':         'balanced_accuracy',
    'seed':            42,
}


class SPDNetClassic(nn.Module):
  
    def __init__(self,
                 n_channels=MODEL_PARAMS['n_channels'],
                 n_classes=MODEL_PARAMS['n_classes'],
                 subspacedim=MODEL_PARAMS['subspacedim']):
        super().__init__()
        self.model = SPDNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            input_type="raw",
            subspacedim=subspacedim,
        )

    def forward(self, x):
        return self.model(x)