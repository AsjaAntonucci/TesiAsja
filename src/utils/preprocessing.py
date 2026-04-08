"""
═══════════════════════════════════════════════════════════════════════════════
                    PREPROCESSING PIPELINE - CHB-MIT EEG DATA
═══════════════════════════════════════════════════════════════════════════════

PIPELINE STEPS:
  1. PARSE INPUT       → Accepts patient ID from command line
  2. LOAD ANNOTATIONS  → Extracts seizure intervals from summary file
  3. CHANNEL SELECTION → Fixed 21 bipolar channels
  4. APPLY FILTERS     → HP 0.5 Hz + LP 80 Hz + Notch 60 Hz Butterworth
  5. ARTIFACT REMOVAL  → Zero samples beyond ±2σ per channel
  6. SEGMENT DATA      → Splits signal into 1-second time windows
  7. LABEL WINDOWS     → Labels as ictal (seizure) or non-ictal
  8. CLASS BALANCING   → Balances classes 4:1 while maintaining temporal order
  9. SAVE TENSOR       → Exports to PyTorch format (.pt)

OUTPUT: .pt file containing tensors X (21, 256) and y (labels)
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import argparse
import warnings
import mne
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt


# ─── Silence MNE non-unique channel warnings ────────────────────────────────
warnings.filterwarnings(
    'ignore',
    message='Channel names are not unique',
    category=RuntimeWarning
)
# ─── [Fix] SEED FOR REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── FIXED 21 BIPOLAR CHANNELS ────────────────────────────────────────────────
BIPOLAR_CHANNELS = [
    'FP1-F7', 'F7-T7',  'T7-P7',   'P7-O1',
    'FP1-F3', 'F3-C3',  'C3-P3',   'P3-O1',
    'FP2-F4', 'F4-C4',  'C4-P4',   'P4-O2',
    'FP2-F8', 'F8-T8',  'T8-P8-0', 'P8-O2',
    'FZ-CZ',  'CZ-PZ',  'T7-FT9',  'FT9-FT10', 'FT10-T8',
]
EXPECTED_CHANNELS = len(BIPOLAR_CHANNELS)  # 21

# ─── STEP 1: PARSE COMMAND LINE ARGUMENTS ─────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Preprocesses CHB-MIT EEG data for seizure detection'
)
parser.add_argument(
    '-p', '--patient',
    type=str,
    required=True,
    help='Patient ID (e.g., chb01 or 01)'
)
args = parser.parse_args()

patient_input = args.patient.strip()
PATIENT_ID = patient_input if patient_input.startswith('chb') else f"chb{patient_input.zfill(2)}"

# ─── CONFIGURATION: PATHS AND PARAMETERS ──────────────────────────────────────
BASE_PATH    = Path("Dataset_CHB-MIT")
DATASET_PATH = BASE_PATH / PATIENT_ID
OUTPUT_DIR   = Path("data") / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 256

# ─── STEP 2: PARSE SUMMARY FILE ───────────────────────────────────────────────
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
print(f"Loaded annotations: {annotations}")

# ─── STEP 3: CHANNEL SELECTION ────────────────────────────────────────────────

def _monopolar_to_bipolar(raw):
    """
    Converts monopolar recording to bipolar.
    Computes V(electrode_A) - V(electrode_B) for each bipolar channel pair.
    """
    data, _ = raw[:, :]
    new_data = np.zeros((EXPECTED_CHANNELS, data.shape[1]))

    for ch_idx, ch_name in enumerate(BIPOLAR_CHANNELS):
        parts        = ch_name.replace('-0', '').split('-')
        elec1, elec2 = parts[0], parts[1]

        flag_1, flag_2 = False, False
        sig1,   sig2   = None,  None

        for i, raw_ch in enumerate(raw.info['ch_names']):
            if 'CP4' in raw_ch:
                continue
            if not flag_1 and elec1 in raw_ch:
                sig1   = data[i, :]
                flag_1 = True
            elif not flag_2 and elec2 in raw_ch:
                sig2   = data[i, :]
                flag_2 = True
            if flag_1 and flag_2:
                new_data[ch_idx, :] = sig1 - sig2
                break

        if not (flag_1 and flag_2):
            print(f"   ⚠ Monopolar: electrodes not found for '{ch_name}'")

    return new_data  # shape: (21, T)


def extract_channels(raw):
    """
    Extracts 21 bipolar channels from EEG recording.
    Handles both bipolar and monopolar recordings.

    Returns:
        np.ndarray shape (21, T), or None if channels cannot be extracted.
    """
    first_ch = raw.info['ch_names'][0]
    is_monopolar = ('-CS' in first_ch) or ('-' not in first_ch)

    if is_monopolar:
        print(f"   → Monopolar recording, converting to bipolar...")
        data = _monopolar_to_bipolar(raw)

    else:
        available = set(raw.ch_names)
        missing   = [ch for ch in BIPOLAR_CHANNELS if ch not in available]
        if missing:
            print(f"   ⚠ Missing channels: {missing} — skip file")
            return None

        raw_sel = raw.copy().pick(BIPOLAR_CHANNELS, verbose=False)

        rename_map = {
            ch: ch.replace('-0', '')
            for ch in raw_sel.ch_names
            if '-0' in ch
        }
        if rename_map:
            raw_sel.rename_channels(rename_map)

        ordered = [ch.replace('-0', '') for ch in BIPOLAR_CHANNELS]
        raw_sel.reorder_channels(ordered)
        data, _ = raw_sel[:, :]

    if data.shape[0] != EXPECTED_CHANNELS:
        print(f"   ⚠ Unexpected shape: {data.shape[0]} channels (expected {EXPECTED_CHANNELS}) — skip")
        return None

    return data  # shape: (21, T)

# ─── STEP 4: SIGNAL FILTERING ─────────────────────────────────────────────────
def highpass(data, cutoff=0.5, fs=FS, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, data, axis=-1)


def lowpass(data, cutoff=80.0, fs=FS, order=2):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data, axis=-1)


def notch(data, freq=60.0, fs=FS, order=2):
    # filtfilt doubles the order → order=2 produces effective order 4 zero-phase
    low  = (freq - 1.0) / (fs / 2)
    high = (freq + 1.0) / (fs / 2)
    b, a = butter(order, [low, high], btype='bandstop')
    return filtfilt(b, a, data, axis=-1)


def preprocess(raw_data):
    """
    HP 0.5 Hz + LP 80 Hz + Notch 60 Hz Butterworth + zeroing ±2σ + z-score.
    """
    x = highpass(raw_data)
    x = lowpass(x)
    x = notch(x)

    # Artifact rejection: zero samples beyond ±2σ per channel
    for ch in range(x.shape[0]):
        mask = np.abs(x[ch] - x[ch].mean()) > 2 * x[ch].std()
        x[ch, mask] = 0.0

    # Z-score normalization: zero mean, unit variance per channel
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return x

# ─── STEP 5: LOAD, PREPROCESS AND SEGMENT ──────────────────────────────────────
def load_and_segment(edf_path, seizure_intervals, fs=FS, window_sec=1):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # ─── Fix MNE duplicates (CHB-MIT: T8-P8 and T8-P8-0 coexist in EDF file) ────
    # MNE auto-renames: T8-P8 → T8-P8-0, T8-P8-0 → T8-P8-1.
    # Strategy: pick excludes T8-P8-0 (unwanted duplicate),
    # then renames T8-P8-1 → T8-P8-0 without conflicts.
    if 'T8-P8-0' in raw.ch_names and 'T8-P8-1' in raw.ch_names:
        ch_to_keep = [ch for ch in raw.ch_names if ch != 'T8-P8-0']
        raw.pick(ch_to_keep, verbose=False)
        raw.rename_channels({'T8-P8-1': 'T8-P8-0'})
    # ─────────────────────────────────────────────────────────────────────────

    data = extract_channels(raw)
    if data is None:
        return None, None

    data = preprocess(data)

    n_samples_window = fs * window_sec
    n_windows        = data.shape[1] // n_samples_window

    windows, labels = [], []
    for i in range(n_windows):
        start    = i * n_samples_window
        end      = start + n_samples_window
        t_start  = start / fs
        t_end    = end   / fs
        is_ictal = any(t_start < e and t_end > s for s, e in seizure_intervals)
        windows.append(data[:, start:end])
        labels.append(1 if is_ictal else 0)

    return np.array(windows), np.array(labels)

# ─── PROCESS ALL PATIENT FILES ────────────────────────────────────────────────
all_windows, all_labels = [], []

all_edfs = sorted(f.name for f in DATASET_PATH.glob("*.edf"))
for edf_file in all_edfs:
    seizure_intervals = annotations.get(edf_file, [])
    edf_path = DATASET_PATH / edf_file
    if not edf_path.exists():
        print(f"   ⚠ File not found, skip: {edf_file}")
        continue
    label = "with seizure" if seizure_intervals else "no seizure"
    print(f"   Process ({label}): {edf_file}")
    X, y = load_and_segment(edf_path, seizure_intervals)
    if X is None:
        continue
    all_windows.append(X)
    all_labels.append(y)

X_all = np.concatenate(all_windows, axis=0)
y_all = np.concatenate(all_labels,  axis=0)

print(f"\nTotal windows: {len(y_all)}")
print(f"  Ictal:     {y_all.sum()}")
print(f"  Non-ictal: {(y_all == 0).sum()}")
print(f"  Ratio:     1:{(y_all==0).sum() // max(y_all.sum(), 1)}")

# ─── STEP 8: CLASS BALANCING 4:1 ──────────────────────────────────────────────
ictal_idx     = np.where(y_all == 1)[0]
non_ictal_idx = np.where(y_all == 0)[0]

n_keep = min(len(non_ictal_idx), 4 * len(ictal_idx))
non_ictal_bal = np.sort(np.random.choice(non_ictal_idx, n_keep, replace=False))
bal_idx = np.sort(np.concatenate([ictal_idx, non_ictal_bal]))

X_bal = X_all[bal_idx]
y_bal = y_all[bal_idx]

print(f"\nAfter 4:1 balancing:")
print(f"  Ictal:     {y_bal.sum()}")
print(f"  Non-ictal: {(y_bal == 0).sum()}")

# ─── STEP 9: SAVE ─────────────────────────────────────────────────────────────
X_tensor = torch.tensor(X_bal, dtype=torch.float32)
y_tensor = torch.tensor(y_bal, dtype=torch.long)

out_path = OUTPUT_DIR / f"{PATIENT_ID}_preprocessed.pt"
torch.save({'X': X_tensor, 'y': y_tensor}, out_path)

print(f"\n[OK] Saved: {out_path}")
print(f"   Shape X: {X_tensor.shape}")
print(f"   Shape y: {y_tensor.shape}")