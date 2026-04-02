import torch
import torch.nn as nn


MODEL_PARAMS = {
    'n_channels': 21,
    'n_classes': 2,

    # DIFFERENZA DA ROSANNA: input shape adattato a finestre da 1s (256 campioni)
    # invece di 4s (1024 campioni). Scelta necessaria per uniformità con SVM e
    # SPDNet nel benchmark: tutti i modelli ricevono lo stesso tensore (batch, 21, 256).
    'input_shape': '(batch, 1, 21, 256)',

    'architecture': '5x [Conv2D(1D-separabile) + BN + ReLU + MaxPool] -> GAP -> Linear',
    'blocks': [
        # DIFFERENZA DA ROSANNA: pool_kernel (1,4) invece di (1,8).
        # Con finestra da 256 campioni, pool(1,8) ridurrebbe troppo aggressivamente
        # la dimensione temporale (256/8=32 già al primo blocco).
        # Con pool(1,4): 256 → 64, coerente con la riduzione progressiva.
        {'in_ch': 1,  'out_ch': 8,  'conv_kernel': [1, 4], 'pool_kernel': [1, 4], 'role': 'filtri temporali'},

        # DIFFERENZA DA ROSANNA: pool_kernel (1,2) invece di (1,4).
        # Stesso motivo: adattamento alla finestra da 256 campioni.
        # 64 → 32 invece di collassare troppo.
        {'in_ch': 8,  'out_ch': 16, 'conv_kernel': [1, 8], 'pool_kernel': [1, 2], 'role': 'filtri temporali'},

        # DIFFERENZA DA ROSANNA: pool_kernel (1,2) invece di (1,4).
        # 32 → 16: si mantiene una risoluzione temporale residua per il GAP finale.
        {'in_ch': 16, 'out_ch': 32, 'conv_kernel': [1, 4], 'pool_kernel': [1, 2], 'role': 'filtri temporali'},

        # DIFFERENZA DA ROSANNA: pool_kernel (21,1) invece di (2,1) o (4,1).
        # Rosanna usa 2 o 4 canali selezionati paziente-specificamente via PCA.
        # Noi usiamo tutti i 21 canali → il pool spaziale deve collassare 21 righe a 1.
        {'in_ch': 32, 'out_ch': 32, 'conv_kernel': [3, 1], 'pool_kernel': [21, 1],'role': 'filtri spaziali'},

        {'in_ch': 32, 'out_ch': 64, 'conv_kernel': [1, 1], 'pool_kernel': [1, 1], 'role': 'compressione finale'},
    ],

    # DIFFERENZA DA ROSANNA: riduzione temporale 256→64→32→16 invece di 1024→128→32→8.
    # Conseguenza diretta dell'uso di finestre da 1s invece di 4s.
    'temporal_reduction': '256 ->div4-> 64 ->div2-> 32 ->div2-> 16 -> spatial collapse -> GAP',

    'gap': 'AdaptiveAvgPool2d(1)',
    'classifier_in': 64,
    'reference': 'Ferrara PhD Thesis (2025) + Sensors 2025 — kernel 1D separabili, pool adattati a 1s/256Hz',
}


TRAIN_PARAMS = {
    'optimizer':   'Adam',
    'lr':          1e-3,
    'weight_decay': 0.0,           # NOTA: disabilitato per uniformità con SPDNet (dataset piccolo)
    'loss':        'CrossEntropyLoss',

    # UGUALE A ROSANNA: max 150 epoche + early stopping con patience=10.
    'epochs':                  150,
    'early_stopping_patience': 10,

    # UGUALE A ROSANNA: batch_size=64.
    'batch_size':  64,

    'train_split':     0.8,
    'val_split_final': 0.9,           # 90% train / 10% val nel training finale
    'cv_folds':        5,
    'cv_strategy':     'StratifiedKFold',
    'seed':            42,
}


class EpiDeNet(nn.Module):
    """
    EpiDeNet — adattato da Ferrara PhD Thesis (2025) + Sensors 2025.

    Differenze rispetto all'originale di Rosanna:
    ─────────────────────────────────────────────
    1. FINESTRA: 1s (256 campioni) invece di 4s (1024 campioni).
       → Uniformità con SVM e SPDNet nel benchmark.

    2. POOL TEMPORALI: ridotti proporzionalmente (÷4, ÷2, ÷2)
       invece di (÷8, ÷4, ÷4) per non collassare troppo
       una finestra già più corta.

    3. CANALI: 21 (tutti i canali bipolari CHB-MIT) invece di
       2–4 selezionati paziente-specificamente via PCA.
       → Il benchmark confronta i modelli a parità di input.
       → Pool spaziale block4: MaxPool(21,1) invece di MaxPool(2,1).

    Invariato rispetto a Rosanna:
    ─────────────────────────────
    - Struttura a 5 blocchi [Conv2D + BN + ReLU + MaxPool]
    - Kernel separabili: block1-3 temporali (1,k), block4 spaziale (3,1)
    - Block5 compressione (1,1) + GAP + Linear classifier
    - Adam lr=1e-3, batch=64, epochs=150, early stopping patience=10

    Input:  (batch, 21, 256)
    Output: (batch, 2)

    Flusso temporale: 256 →÷4→ 64 →÷2→ 32 →÷2→ 16 → GAP
    Flusso spaziale:  21 canali collassati a 1 in block4
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
        x = x.unsqueeze(1)                          # (batch, 21, 256) → (batch, 1, 21, 256)
        x = self.block1(x)                          # → (batch,  8, 21,  64)   tempo: 256÷4=64
        x = self.block2(x)                          # → (batch, 16, 21,  32)   tempo: 64÷2=32
        x = self.block3(x)                          # → (batch, 32, 21,  16)   tempo: 32÷2=16
        x = self.block4(x)                          # → (batch, 32,  1,  16)   spazio: 21→1
        x = self.block5(x)                          # → (batch, 64,  1,  16)   compressione
        x = self.gap(x).squeeze(-1).squeeze(-1)     # → (batch, 64)
        return self.classifier(x)                   # → (batch, 2)
