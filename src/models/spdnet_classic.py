import torch.nn as nn
from spd_learn.modules import CovLayer, BiMap, ReEig, LogEig, Shrinkage


MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,
    'init_shrinkage': 0.1,
    'bimap1_out': 10,
    'bimap2_out': 5,
    'bimap3_out': 2,
    'tangent_dim': 3,
    'architecture': 'CovLayer → Shrinkage → BiMap → ReEig → BiMap → ReEig → BiMap → ReEig → LogEig → Linear',
    'reference': 'Huang & Van Gool (AAAI 2017)',
}

TRAIN_PARAMS = {
    'optimizer':       'Adam',
    'lr':              1e-3,
    'weight_decay':    0.0,
    'loss':            'CrossEntropyLoss',
    'epochs':          500,
    'patience':        25,
    'batch_size':      32,
    'train_split':     0.8,
    'val_split_final': 0.9,
    'cv_folds':        5,
    'cv_strategy':     'StratifiedKFold',
    'scoring':         'balanced_accuracy',
    'seed':            42,
}


class SPDNetClassic(nn.Module):
    """
    SPDNet classico — Huang & Van Gool (AAAI 2017)
    Pipeline: Raw → CovLayer → Shrinkage → BiMap → ReEig → BiMap → ReEig → BiMap → ReEig → LogEig → Linear
    Input: (batch, 21, 256)
    Output: (batch, 2)
    """
    def __init__(self, n_channels=MODEL_PARAMS['n_channels'],
                 n_classes=MODEL_PARAMS['n_classes']):
        super().__init__()
        self.cov = CovLayer()
        self.shrinkage = Shrinkage(n_chans=n_channels,
                                   init_shrinkage=MODEL_PARAMS['init_shrinkage'])
        self.bimap1 = BiMap(in_features=n_channels,
                            out_features=n_channels // 2)   # 21 → 10
        self.reeig1 = ReEig()
        self.bimap2 = BiMap(in_features=n_channels // 2,
                            out_features=n_channels // 4)   # 10 → 5
        self.reeig2 = ReEig()
        self.bimap3 = BiMap(in_features=n_channels // 4,
                            out_features=n_channels // 8)   # 5 → 2
        self.reeig3 = ReEig()
        self.logeig = LogEig(upper=True)
        tangent_dim = (n_channels // 8) * (n_channels // 8 + 1) // 2  # 2*3//2 = 3
        self.classifier = nn.Linear(tangent_dim, n_classes)

    def forward(self, x):
        x = self.cov(x)
        x = self.shrinkage(x)
        x = self.bimap1(x)
        x = self.reeig1(x)
        x = self.bimap2(x)
        x = self.reeig2(x)
        x = self.bimap3(x)
        x = self.reeig3(x)
        x = self.logeig(x)
        return self.classifier(x)
