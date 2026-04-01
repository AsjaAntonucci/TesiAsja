import torch
import torch.nn as nn

MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,
    'input_shape': '(batch, 1, 21, 256)',
    'architecture': '5x [Conv2D(1D-separabile) + BN + ReLU + MaxPool] -> GAP -> Linear',
    'blocks': [
        {'in_ch': 1,  'out_ch': 8,  'conv_kernel': [1, 4], 'pool_kernel': [1, 4], 'role': 'filtri temporali'},
        {'in_ch': 8,  'out_ch': 16, 'conv_kernel': [1, 8], 'pool_kernel': [1, 2], 'role': 'filtri temporali'},
        {'in_ch': 16, 'out_ch': 32, 'conv_kernel': [1, 4], 'pool_kernel': [1, 2], 'role': 'filtri temporali'},
        {'in_ch': 32, 'out_ch': 32, 'conv_kernel': [3, 1], 'pool_kernel': [21, 1],'role': 'filtri spaziali'},
        {'in_ch': 32, 'out_ch': 64, 'conv_kernel': [1, 1], 'pool_kernel': [1, 1], 'role': 'compressione finale'},
    ],
    'temporal_reduction': '256 ->div4-> 64 ->div2-> 32 ->div2-> 16 -> spatial collapse -> GAP',
    'gap': 'AdaptiveAvgPool2d(1)',
    'classifier_in': 64,
    'reference': 'Ferrara PhD Thesis (2025) + Sensors 2025 — kernel 1D separabili, pool adattati a 1s/256Hz',
}

TRAIN_PARAMS = {
    'optimizer':                'Adam',
    'lr':                       1e-3,
    'weight_decay':             1e-4,
    'loss':                     'CrossEntropyLoss',
    'epochs':                   150,          # era 50 — allineato a Rosanna (max 150 + early stopping)
    'batch_size':               64,           # era 32 — allineato a Rosanna
    'train_split':              0.8,
    'val_split_final':          0.9,          # 90% train / 10% val nel training finale
    'cv_folds':                 5,
    'cv_strategy':              'StratifiedKFold',
    'early_stopping_patience':  10,           # nuovo — allineato a Rosanna
    'seed':                     42,
}


class EpiDeNet(nn.Module):
    """
    EpiDeNet — adattato da Ferrara PhD Thesis (2025) + Sensors 2025
    Kernel 1D separabili: block1-3 operano sul tempo, block4 sui canali.
    Pool adattati a finestre da 1s (256 campioni) invece di 4s (1024 campioni).
    Input:  (batch, 21, 256)
    Output: (batch, 2)

    Flusso temporale: 256 ->div4-> 64 ->div2-> 32 ->div2-> 16 -> GAP
    Flusso spaziale:  21 canali collassati a 1 nel block4 con MaxPool(21,1)
    """
    def __init__(self, n_channels=MODEL_PARAMS['n_channels'],
                 n_classes=MODEL_PARAMS['n_classes']):
        super().__init__()
        # Blocchi temporali — kernel (1,k): guardano SOLO il tempo
        self.block1 = self._make_block(1,  8,  conv_kernel=(1, 4), padding=(0, 2), pool_kernel=(1, 4))
        self.block2 = self._make_block(8,  16, conv_kernel=(1, 8), padding=(0, 4), pool_kernel=(1, 2))
        self.block3 = self._make_block(16, 32, conv_kernel=(1, 4), padding=(0, 2), pool_kernel=(1, 2))
        # Blocco spaziale — kernel (3,1): guarda SOLO i canali
        self.block4 = self._make_block(32, 32, conv_kernel=(3, 1), padding=(1, 0), pool_kernel=(n_channels, 1))
        # Compressione finale
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
        x = x.unsqueeze(1)                          # (batch, 1,  21, 256) per trattare i canali come "altezza" e il tempo come "larghezza"
        x = self.block1(x)                           # (batch, 8,  21,  64)
        x = self.block2(x)                           # (batch, 16, 21,  32)
        x = self.block3(x)                           # (batch, 32, 21,  16)
        x = self.block4(x)                           # (batch, 32,  1,  16)
        x = self.block5(x)                           # (batch, 64,  1,  16)
        x = self.gap(x).squeeze(-1).squeeze(-1)      # (batch, 64)
        return self.classifier(x)                    # (batch, 2)
