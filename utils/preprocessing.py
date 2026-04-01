"""
═══════════════════════════════════════════════════════════════════════════════
                    PREPROCESSING PIPELINE - CHB-MIT EEG DATA
═══════════════════════════════════════════════════════════════════════════════

PIPELINE STEPS:
  1. PARSE INPUT       → Accetta ID del paziente da linea di comando
  2. LOAD ANNOTATIONS  → Estrae gli intervalli delle crisi dal file summary
  3. APPLY FILTERS     → Highpass, Lowpass, Notch per pulizia del segnale
  4. SEGMENT DATA      → Divide il segnale in finestre temporali di 1 secondo
  5. LABEL WINDOWS     → Etichetta come ictale (crisi) o non-ictale
  6. CLASS BALANCING   → Bilancia le classi 4:1 (non-ictale : ictale)
  7. SAVE TENSOR       → Esporta in formato PyTorch (.pt)

OUTPUT: File .pt contenente tensori X (segnali) e y (etichette)
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import argparse
import mne
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch


# ─── STEP 1: PARSE ARGOMENTI DA RIGA DI COMANDO ────────────────────────────────
parser = argparse.ArgumentParser(
    description='Preprocessa i dati EEG del paziente CHB-MIT'
)
parser.add_argument(
    '-p', '--patient',
    type=str,
    required=True,
    help='ID del paziente (es. chb01 oppure 01)'
)
args = parser.parse_args()

patient_input = args.patient.strip()
if patient_input.startswith('chb'):
    PATIENT_ID = patient_input
else:
    PATIENT_ID = f"chb{patient_input.zfill(2)}"

# ─── CONFIGURAZIONE PERCORSI E PARAMETRI ──────────────────────────────────────
BASE_PATH    = Path("Dataset_CHB-MIT")
DATASET_PATH = BASE_PATH / PATIENT_ID
OUTPUT_DIR   = Path("data") / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 256
EXPECTED_CHANNELS = 21


# ─── STEP 2: PARSING DEL FILE DI SUMMARY ───────────────────────────────────────
def parse_summary(summary_path):
    seizures = {}
    current_file = None
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'File Name:\s+(\S+\.edf)', line)
            if m:
                current_file = m.group(1)
                seizures[current_file] = []
            start_m = re.match(r'Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds', line)
            end_m   = re.match(r'Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds', line)
            if start_m and current_file:
                seizures[current_file].append([int(start_m.group(1)), None])
            if end_m and current_file and seizures[current_file]:
                seizures[current_file][-1][1] = int(end_m.group(1))
    return {k: [(s, e) for s, e in v if e is not None]
            for k, v in seizures.items()}


summary_path = DATASET_PATH / f"{PATIENT_ID}-summary.txt"
annotations  = parse_summary(summary_path)
print(f"Annotazioni caricate: {annotations}")


# ─── STEP 3: FUNZIONI DI FILTRAGGIO DEL SEGNALE ────────────────────────────────
def highpass(data, cutoff=0.5, fs=256, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, data, axis=-1)

def lowpass(data, cutoff=80.0, fs=256, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data, axis=-1)

def notch(data, freq=60.0, fs=256, quality=30):
    b, a = iirnotch(freq / (fs / 2), quality)
    return filtfilt(b, a, data, axis=-1)

def preprocess(raw_data):
    x = highpass(raw_data)
    x = lowpass(x)
    x = notch(x)
    std  = x.std(axis=-1, keepdims=True)
    mean = x.mean(axis=-1, keepdims=True)
    mask = np.any(np.abs(x - mean) > 2 * std, axis=0)
    x[:, mask] = 0
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return x


# ─── STEP 4: CARICAMENTO, PREPROCESSING E SEGMENTAZIONE ────────────────────────
def load_and_segment(edf_path, seizure_intervals, fs=256, window_sec=1):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch or '-' in ch][:21]

    # ✅ FIX chb12: salta i file con meno di 21 canali EEG validi
    if len(eeg_channels) < EXPECTED_CHANNELS:
        print(f"   ⚠ Skip {edf_path.name}: solo {len(eeg_channels)} canali EEG (attesi {EXPECTED_CHANNELS})")
        return None, None

    raw.pick(eeg_channels)
    data = raw.get_data()
    data = preprocess(data)

    n_samples_window = fs * window_sec
    n_windows = data.shape[1] // n_samples_window

    windows, labels = [], []
    for i in range(n_windows):
        start   = i * n_samples_window
        end     = start + n_samples_window
        t_start = start / fs
        t_end   = end   / fs
        is_ictal = any(t_start < e and t_end > s for s, e in seizure_intervals)
        windows.append(data[:, start:end])
        labels.append(1 if is_ictal else 0)

    return np.array(windows), np.array(labels)


# ─── STEP 5: PROCESSAMENTO DI TUTTI I FILE DEL PAZIENTE ────────────────────────
all_windows, all_labels = [], []

# Processa i file con crisi annotate
for edf_file, seizure_intervals in annotations.items():
    edf_path = DATASET_PATH / edf_file
    if not edf_path.exists():
        print(f"   ⚠ File non trovato, skip: {edf_file}")
        continue
    print(f"   Processo (con crisi): {edf_file}")
    X, y = load_and_segment(edf_path, seizure_intervals)
    if X is None:   # ✅ FIX: skip file con canali insufficienti
        continue
    all_windows.append(X)
    all_labels.append(y)

# Processa i file senza crisi annotate
all_edfs = [f.name for f in DATASET_PATH.glob("*.edf")]
for edf_file in all_edfs:
    if edf_file not in annotations:
        edf_path = DATASET_PATH / edf_file
        print(f"   Processo (senza crisi): {edf_file}")
        X, y = load_and_segment(edf_path, seizure_intervals=[])
        if X is None:   # ✅ FIX: skip file con canali insufficienti
            continue
        all_windows.append(X)
        all_labels.append(y)

X_all = np.concatenate(all_windows, axis=0)
y_all = np.concatenate(all_labels,  axis=0)

print(f"\nFinestre totali: {len(y_all)}")
print(f"  Ictali:     {y_all.sum()}")
print(f"  Non-ictali: {(y_all == 0).sum()}")
print(f"  Rapporto:   1:{(y_all==0).sum() // max(y_all.sum(), 1)}")


# ─── STEP 6: CLASS BALANCING (RAPPORTO 4:1) ────────────────────────────────────
ictal_idx     = np.where(y_all == 1)[0]
non_ictal_idx = np.where(y_all == 0)[0]

n_keep = min(len(non_ictal_idx), 4 * len(ictal_idx))
non_ictal_idx_balanced = np.random.choice(non_ictal_idx, n_keep, replace=False)

balanced_idx = np.concatenate([ictal_idx, non_ictal_idx_balanced])
np.random.shuffle(balanced_idx)

X_bal = X_all[balanced_idx]
y_bal = y_all[balanced_idx]


# ─── STEP 7: SALVATAGGIO COME TENSORI PYTORCH ─────────────────────────────────
X_tensor = torch.tensor(X_bal, dtype=torch.float32)
y_tensor = torch.tensor(y_bal, dtype=torch.long)

out_path = OUTPUT_DIR / f"{PATIENT_ID}_preprocessed.pt"
torch.save({'X': X_tensor, 'y': y_tensor}, out_path)

print(f"\n✅ Salvato: {out_path}")
print(f"   Shape X: {X_tensor.shape}")
print(f"   Shape y: {y_tensor.shape}")
