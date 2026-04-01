import numpy as np
from scipy.signal import welch
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
    'gamma':        'scale',
    'class_weight': 'balanced',
    'probability':  False,
}

FEATURE_PARAMS = {
    'n_channels':  21,
    'n_bands':     8,
    'n_features':  21 * 8 * W,  # 504
    'fs':          256,
    'nperseg':     256,
    'W':           W,
    'bands':       BANDS,
}

TRAIN_PARAMS = {
    'train_split':    0.8,
    'cv_folds':       5,
    'cv_strategy':    'StratifiedKFold',
    'scoring':        'balanced_accuracy',
    'scaler':         'StandardScaler',
}

def extract_features_single(X, fs=FEATURE_PARAMS['fs']):
    """Extract spectral features for each sample (no temporal stacking)."""
    N, n_channels = X.shape[0], X.shape[1]
    n_bands = len(BANDS)
    features = np.zeros((N, n_channels * n_bands))
    for i in range(N):
        for ch in range(n_channels):
            freqs, psd = welch(X[i, ch], fs=fs, nperseg=fs)
            for b, (_, (fmin, fmax)) in enumerate(BANDS.items()):
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                features[i, ch * n_bands + b] = np.sum(psd[idx])
    return features

def extract_features(X, fs=FEATURE_PARAMS['fs'], W=FEATURE_PARAMS['W']):
    """Extract spectral features with temporal stacking (window size W)."""
    feat = extract_features_single(X, fs)
    N = len(feat)
    return np.array([feat[i:i + W].flatten() for i in range(N - W + 1)])

def build_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(**SVM_PARAMS))
    ])
