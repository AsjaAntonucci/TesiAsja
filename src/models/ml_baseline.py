import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


_edges = np.linspace(0.5, 25, 9)
BANDS = {f'band{i+1}': (_edges[i], _edges[i+1]) for i in range(8)}


W = 3  # temporal stacking (guarda tre finestre consecutive per stabilizzare le feature)


# ─── Parametri esposti esplicitamente ────────────────────────────────────────
SVM_PARAMS = {
    'kernel':       'rbf',
    'C':            1.0,

    # DIFFERENZA DAL PAPER: Shoeb & Guttag usano gamma=0.1 fisso, calibrato su segnali
    # grezzi in µV². Il nostro preprocessing applica z-score per canale prima
    # dell'estrazione delle feature, rendendo i valori adimensionali e le distanze
    # nel feature space non comparabili con quelle del paper.
    # gamma='scale' calcola automaticamente gamma = 1 / (n_features * X.var()),
    # adattandosi alla scala reale dei dati dopo normalizzazione.
    'gamma':        'scale',

    # DIFFERENZA DAL PAPER: Shoeb & Guttag bilanciano le classi tramite
    # sottocampionamento esplicito nel training set (seizure vs 24h non-seizure).
    # Noi usiamo class_weight='balanced' per gestire lo sbilanciamento residuo
    # dopo il bilanciamento 4:1 nel preprocessing.
    'class_weight': 'balanced',

    'probability':  False,
}


FEATURE_PARAMS = {
    'n_channels':  21,    # DIFFERENZA DAL PAPER: 18 canali → noi 21 canali bipolari (CHB-MIT completo)
    'n_bands':     8,
    'n_features':  21 * 8 * W,  # 504 (paper: 18*8*3 = 432)
    'fs':          256,
    'W':           W,
    'bands':       BANDS,
    'filter_order': 2,
}


TRAIN_PARAMS = {
    'train_split':    0.8,
    'cv_folds':       5,
    'cv_strategy':    'StratifiedKFold',
    'scoring':        'balanced_accuracy',

    # DIFFERENZA DAL PAPER: il paper non usa StandardScaler perché lavora su
    # segnali grezzi in µV². Noi applichiamo z-score nel preprocessing (condiviso
    # con tutti i modelli del benchmark), quindi le energie estratte sono
    # adimensionali. StandardScaler normalizza le feature fittando SOLO sul
    # training set, evitando data leakage verso il test set.
    'scaler':         'StandardScaler',
}


def _make_filterbank(bands, fs, order=2):
    """Pre-costruisce i filtri Butterworth per ogni banda."""
    nyq = fs / 2.0
    return [
        butter(order, [flo / nyq, fhi / nyq], btype='bandpass', output='sos')
        for (flo, fhi) in bands.values()
    ]


_FILTERBANK = _make_filterbank(BANDS, FEATURE_PARAMS['fs'], order=FEATURE_PARAMS['filter_order'])


def extract_features_single(X, fs=FEATURE_PARAMS['fs']):
    """
    Filterbank energy — fedele a Shoeb & Guttag 2010.
    Per ogni canale, per ogni banda: filtra nel tempo e misura mean(x²).
    """
    N, C = X.shape[0], X.shape[1]
    B = len(_FILTERBANK)
    features = np.zeros((N, C * B))
    for i in range(N):
        for ch in range(C):
            for b, sos in enumerate(_FILTERBANK):
                filtered = sosfilt(sos, X[i, ch])
                features[i, ch * B + b] = np.mean(filtered ** 2)
    return features


def extract_features(X, fs=FEATURE_PARAMS['fs'], W=FEATURE_PARAMS['W']):
    """Temporal stacking su W finestre consecutive."""
    feat = extract_features_single(X, fs)
    N = len(feat)
    return np.array([feat[i:i + W].flatten() for i in range(N - W + 1)])


def build_model():
    """
    SVM pipeline: StandardScaler + RBF-SVM.

    Differenze rispetto a Shoeb & Guttag 2010:
    - gamma='scale' invece di 0.1: necessario perché il preprocessing applica
      z-score al segnale, rendendo le feature adimensionali (non in µV²).
    - StandardScaler: garantisce normalizzazione senza data leakage (fit su train only).
    - 21 canali invece di 18: copertura completa del montaggio CHB-MIT.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(**SVM_PARAMS))
    ])