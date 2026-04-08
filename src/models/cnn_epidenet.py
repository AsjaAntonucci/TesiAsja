"""
═══════════════════════════════════════════════════════════════════════════════
                    MODEL - EpiDeNet CNN Architecture
═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE:
  5 convolutional blocks with separable kernels:
  - Block 1-3: Temporal kernels (1,k) - process time dimension
  - Block 4: Spatial kernel (3,1) - process channel dimension
  - Block 5: Compression (1,1) + Global Average Pool + Linear classifier

DESIGN:
  Input:  (batch, 21, 256)  - 21 bipolar channels, 256 temporal samples (1 second @ 256Hz)
  Output: (batch, 2)        - Binary classification (seizure/non-seizure)
  
  Temporal reduction: 256 → 64 → 32 → 16 (via MaxPool)
  Spatial reduction: 21 channels → 1 (MaxPool in block 4)
  Final features: 64 channels processed by linear classifier

REFERENCE:
  Based on Ferrara PhD Thesis (2025) + Sensors Journal 2025
  Adapted for CHB-MIT dataset: 21 channels, 1-second windows
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn


MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,
    'input_shape': '(batch, 1, 21, 256)',
    'architecture': '5x [Conv2D(1D-separabile) + BN + ReLU + MaxPool] -> GAP -> Linear',
    'blocks': [
        {'in_ch': 1,  'out_ch': 8,  'conv_kernel': [1, 4], 'pool_kernel': [1, 4], 'role': 'temporal filters'},
        {'in_ch': 8,  'out_ch': 16, 'conv_kernel': [1, 8], 'pool_kernel': [1, 2], 'role': 'temporal filters'},
        {'in_ch': 16, 'out_ch': 32, 'conv_kernel': [1, 4], 'pool_kernel': [1, 2], 'role': 'temporal filters'},
        {'in_ch': 32, 'out_ch': 32, 'conv_kernel': [3, 1], 'pool_kernel': [21, 1],'role': 'spatial filters'},
        {'in_ch': 32, 'out_ch': 64, 'conv_kernel': [1, 1], 'pool_kernel': [1, 1], 'role': 'final compression'},
    ],
    'temporal_reduction': '256 ->div4-> 64 ->div2-> 32 ->div2-> 16 -> spatial collapse -> GAP',
    'gap': 'AdaptiveAvgPool2d(1)',
    'classifier_in': 64,
}


TRAIN_PARAMS = {
    'optimizer':   'Adam',
    'lr':           1e-3,
    'weight_decay': 1e-4,
    'loss':        'CrossEntropyLoss',
    'epochs':                  150,
    'early_stopping_patience': 10,
    'batch_size':  64,
    'train_split':     0.8,
    'val_split_final': 0.9,
    'cv_folds':        5,
    'cv_strategy':     'StratifiedKFold',
    'seed':            42,
}


class EpiDeNet(nn.Module):

    def __init__(self, n_channels=MODEL_PARAMS['n_channels'],
                 n_classes=MODEL_PARAMS['n_classes']):
        super().__init__()
        self.block1 = self._make_block(1,  8,  conv_kernel=(1, 4), padding=(0, 2), pool_kernel=(1, 4))
        self.block2 = self._make_block(8,  16, conv_kernel=(1, 8), padding=(0, 4), pool_kernel=(1, 2))
        self.block3 = self._make_block(16, 32, conv_kernel=(1, 4), padding=(0, 2), pool_kernel=(1, 2))
        self.block4 = self._make_block(32, 32, conv_kernel=(3, 1), padding=(1, 0), pool_kernel=(n_channels, 1))
        self.block5 = self._make_block(32, 64, conv_kernel=(1, 1), padding=(0, 0), pool_kernel=(1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    @staticmethod
    def _make_block(in_ch, out_ch, conv_kernel, padding, pool_kernel):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=conv_kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
