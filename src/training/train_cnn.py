import argparse
import copy
import csv
import json
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cnn_epidenet import EpiDeNet, MODEL_PARAMS, TRAIN_PARAMS


SEED = TRAIN_PARAMS['seed']
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--patient', type=str, required=True)
args = parser.parse_args()
patient_input = args.patient.strip()
PATIENT_ID = patient_input if patient_input.startswith('chb') else f"chb{patient_input.zfill(2)}"


RESULTS_DIR    = Path("results") / "metrics"
CHECKPOINT_DIR = Path("results") / "checkpoints"
PARAMS_DIR     = Path("results") / "params"
PLOTS_DIR      = Path("results") / "plots"
for d in [RESULTS_DIR, CHECKPOINT_DIR, PARAMS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


data_path = Path("data") / "preprocessed" / f"{PATIENT_ID}_preprocessed.pt"
data = torch.load(data_path)
X, y = data['X'], data['y']
print(f"✅ Dati caricati — Shape X: {X.shape}, y: {y.shape}")


# ─── Train / Test split 80/20 temporale ──────────────────────────────────────
split = int(len(y) * TRAIN_PARAMS['train_split'])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

MAX_EPOCHS = TRAIN_PARAMS['epochs']
PATIENCE   = TRAIN_PARAMS['early_stopping_patience']


# ─── 5-fold Cross-Validation (StratifiedKFold) ────────────────────────────
print(f"\nCross-Validation ({TRAIN_PARAMS['cv_folds']}-fold stratified)...")
kf = StratifiedKFold(n_splits=TRAIN_PARAMS['cv_folds'], shuffle=False)
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    Xf_tr, Xf_val = X_train[tr_idx], X_train[val_idx]
    yf_tr, yf_val = y_train[tr_idx], y_train[val_idx]

    fold_loader = DataLoader(TensorDataset(Xf_tr, yf_tr),
                             batch_size=TRAIN_PARAMS['batch_size'],
                             shuffle=True)
    fold_model = EpiDeNet(n_channels=MODEL_PARAMS['n_channels'],
                          n_classes=MODEL_PARAMS['n_classes'])
    fold_opt   = torch.optim.Adam(fold_model.parameters(),
                                  lr=TRAIN_PARAMS['lr'],
                                  weight_decay=TRAIN_PARAMS['weight_decay'])
    fold_crit  = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_fold_state = None
    wait = 0

    for epoch in range(MAX_EPOCHS):
        fold_model.train()
        for Xb, yb in fold_loader:
            fold_opt.zero_grad()
            fold_crit(fold_model(Xb), yb).backward()
            fold_opt.step()

        fold_model.eval()
        with torch.no_grad():
            val_loss = fold_crit(fold_model(Xf_val), yf_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_fold_state = copy.deepcopy(fold_model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Fold {fold}: early stopping @ epoch {epoch+1}")
                break

    fold_model.load_state_dict(best_fold_state)
    fold_model.eval()
    with torch.no_grad():
        preds = fold_model(Xf_val).argmax(dim=1).numpy()
    score = balanced_accuracy_score(yf_val.numpy(), preds)
    cv_scores.append(round(float(score), 4))
    print(f"  Fold {fold}: bAcc = {score:.4f}")

cv_scores = np.array(cv_scores)
print(f"  Media CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─── Grafico CV scores ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
folds_x = list(range(1, len(cv_scores) + 1))
ax.bar(folds_x, cv_scores, color='darkorange', alpha=0.8, label='bAcc per fold')
ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Media = {cv_scores.mean():.4f}')
ax.fill_between(folds_x,
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color='red', label=f'±1 std = {cv_scores.std():.4f}')
ax.set_xlabel('Fold'); ax.set_ylabel('Balanced Accuracy')
ax.set_title(f'EpiDeNet Cross-Validation — {PATIENT_ID}')
ax.set_ylim(0, 1.05); ax.set_xticks(folds_x); ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{PATIENT_ID}_cnn_cv.png", dpi=150); plt.close()
print(f"  📈 Grafico CV salvato")


# ─── Training finale ──────────────────────────────────────────────────────────
print("\nTraining finale EpiDeNet...")

# Split temporale 90/10 dentro X_train per early stopping
val_split = int(len(y_train) * TRAIN_PARAMS['val_split_final'])
X_tr, X_val_fin = X_train[:val_split], X_train[val_split:]
y_tr, y_val_fin = y_train[:val_split], y_train[val_split:]

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=TRAIN_PARAMS['batch_size'],
                          shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test),
                          batch_size=TRAIN_PARAMS['batch_size'], shuffle=False)

model     = EpiDeNet(n_channels=MODEL_PARAMS['n_channels'],
                     n_classes=MODEL_PARAMS['n_classes'])
optimizer = torch.optim.Adam(model.parameters(),
                             lr=TRAIN_PARAMS['lr'],
                             weight_decay=TRAIN_PARAMS['weight_decay'])
criterion = nn.CrossEntropyLoss()
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parametri totali: {n_params:,}")

train_loss_history = []
val_loss_history   = []
best_val_loss  = float('inf')
best_state     = None
wait           = 0
stopped_epoch  = MAX_EPOCHS

for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_loss_history.append(round(avg_train_loss, 6))

    model.eval()
    with torch.no_grad():
        avg_val_loss = criterion(model(X_val_fin), y_val_fin).item()
    val_loss_history.append(round(avg_val_loss, 6))

    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            stopped_epoch = epoch + 1
            print(f"  Early stopping @ epoch {stopped_epoch}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} — Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

model.load_state_dict(best_state)
print(f"  ✅ Best model caricato (best_val_loss={best_val_loss:.6f})")


# ─── Grafico loss curve (train + val) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
epochs_x = range(1, len(train_loss_history) + 1)
ax.plot(epochs_x, train_loss_history, color='darkorange', linewidth=1.8, label='Train Loss')
ax.plot(epochs_x, val_loss_history,   color='steelblue',  linewidth=1.8,
        linestyle='--', label='Val Loss')
ax.axvline(stopped_epoch, color='red', linestyle=':', linewidth=1.2,
           label=f'Early stop @ ep.{stopped_epoch}')
ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy Loss')
ax.set_title(f'EpiDeNet Training — {PATIENT_ID}')
ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(PLOTS_DIR / f"{PATIENT_ID}_cnn_loss.png", dpi=150); plt.close()
print(f"  📈 Loss curve salvata")


# ─── Valutazione sul test set ─────────────────────────────────────────────────
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = model(Xb).argmax(dim=1)
        all_preds.extend(preds.numpy()); all_true.extend(yb.numpy())

bAcc = balanced_accuracy_score(all_true, all_preds)
tn, fp, fn, tp = confusion_matrix(all_true, all_preds, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n✅ Risultati {PATIENT_ID} — EpiDeNet CNN:")
print(f"  Balanced Accuracy : {bAcc:.4f}")
print(f"  Sensitivity       : {sensitivity:.4f}")
print(f"  Specificity       : {specificity:.4f}")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"  CV Media          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Early stop epoch  : {stopped_epoch}")


# ─── Salva parametri JSON ─────────────────────────────────────────────────────
params_log = {
    'patient': PATIENT_ID, 'model': 'EpiDeNet_CNN',
    'model_params': MODEL_PARAMS, 'train_params': TRAIN_PARAMS,
    'n_params': n_params,
    'n_train': int(len(X_tr)), 'n_val': int(len(X_val_fin)), 'n_test': int(len(X_test)),
    'n_ictal_train': int(y_tr.sum().item()),
    'n_ictal_val':   int(y_val_fin.sum().item()),
    'n_ictal_test':  int(y_test.sum().item()),
    'cv_scores':          [round(float(s), 4) for s in cv_scores],
    'cv_mean':            round(float(cv_scores.mean()), 4),
    'cv_std':             round(float(cv_scores.std()),  4),
    'train_loss_history': train_loss_history,
    'val_loss_history':   val_loss_history,
    'best_val_loss':      round(best_val_loss, 6),
    'stopped_epoch':      stopped_epoch,
    'test_bAcc':          round(float(bAcc), 4),
    'sensitivity':        round(float(sensitivity), 4),
    'specificity':        round(float(specificity), 4),
}
with open(PARAMS_DIR / f"{PATIENT_ID}_cnn_params.json", 'w') as f:
    json.dump(params_log, f, indent=2)
print(f"  🔧 Parametri salvati")


# ─── Salva risultati CSV (schema unificato benchmark) ─────────────────────────
csv_path = RESULTS_DIR / "benchmark_results.csv"
write_header = not csv_path.exists()
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model", "patient", "bAcc", "sensitivity", "specificity",
        "TP", "FP", "TN", "FN", "cv_mean", "cv_std", "final_loss", "stopped_epoch"])
    if write_header:
        writer.writeheader()
    writer.writerow({
        "model":         "EpiDeNet_CNN",
        "patient":       PATIENT_ID,
        "bAcc":          round(bAcc, 4),
        "sensitivity":   round(sensitivity, 4),
        "specificity":   round(specificity, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "cv_mean":       round(float(cv_scores.mean()), 4),
        "cv_std":        round(float(cv_scores.std()),  4),
        "final_loss":    round(best_val_loss, 6),   # best val loss, non train loss
        "stopped_epoch": stopped_epoch,
    })
print(f"  📊 Risultati salvati in benchmark_results.csv")


# ─── Salva best model ─────────────────────────────────────────────────────────
torch.save(model.state_dict(), CHECKPOINT_DIR / f"{PATIENT_ID}_cnn_epidenet.pt")
print(f"  💾 Best model salvato")