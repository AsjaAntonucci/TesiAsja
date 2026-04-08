"""
═══════════════════════════════════════════════════════════════════════════════
                    MODEL - SVM Baseline with Spectral Features
═══════════════════════════════════════════════════════════════════════════════

FEATURE EXTRACTION:
  1. Filterbank: 8 bandpass Butterworth filters (0.5-25 Hz) - Shoeb & Guttag 2010
  2. Energy computation: mean(filtered_signal^2) per channel per band
  3. Temporal stacking: W=3 consecutive windows concatenated
  4. Output: 504 features (21 channels * 8 bands * 3 windows)

CLASSIFIER:
  SVM with RBF kernel, gamma='scale', balanced class weights
  Pipeline: StandardScaler → SVC

RESEULT:
  Baseline reference model from literature implemented for benchmark
  Reference: Shoeb & Guttag, IEEE Trans Biomed Eng 2010
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


_edges = np.linspace(0.5, 25, 9)
BANDS = {f'band{i+1}': (_edges[i], _edges[i+1]) for i in range(8)}

W = 3


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
    'n_features':  21 * 8 * W,
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
    'scaler':         'StandardScaler',
}


def _make_filterbank(bands, fs, order=2):
    """Pre-constructs Butterworth filters for each frequency band."""
    nyq = fs / 2.0
    return [
        butter(order, [flo / nyq, fhi / nyq], btype='bandpass', output='sos')
        for (flo, fhi) in bands.values()
    ]


_FILTERBANK = _make_filterbank(BANDS, FEATURE_PARAMS['fs'], order=FEATURE_PARAMS['filter_order'])


def extract_features_single(X, fs=FEATURE_PARAMS['fs']):
    """Filterbank energy - spectral power computation per channel per band."""
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
    """Temporal stacking of W consecutive windows."""
    feat = extract_features_single(X, fs)
    N = len(feat)
    return np.array([feat[i:i + W].flatten() for i in range(N - W + 1)])


def build_model():
    """Create SVM pipeline with StandardScaler."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(**SVM_PARAMS))
    ])