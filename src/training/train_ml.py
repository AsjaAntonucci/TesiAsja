import argparse
import csv
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.ml_baseline import extract_features, build_model, SVM_PARAMS, FEATURE_PARAMS, TRAIN_PARAMS

# ─── Argomenti ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--patient', type=str, required=True)
args = parser.parse_args()
patient_input = args.patient.strip()
PATIENT_ID = patient_input if patient_input.startswith('chb') else f"chb{patient_input.zfill(2)}"

# ─── Cartelle output ──────────────────────────────────────────────────────────
RESULTS_DIR    = Path("results") / "metrics"
CHECKPOINT_DIR = Path("results") / "checkpoints"
PARAMS_DIR     = Path("results") / "params"
PLOTS_DIR      = Path("results") / "plots"
for d in [RESULTS_DIR, CHECKPOINT_DIR, PARAMS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Carica dati ──────────────────────────────────────────────────────────────
data_path = Path("data") / "preprocessed" / f"{PATIENT_ID}_preprocessed.pt"
data = torch.load(data_path)
X = data['X'].numpy()
y = data['y'].numpy()
print(f"✅ Dati caricati — Shape X: {X.shape}, y: {y.shape}")

# ─── Feature extraction ───────────────────────────────────────────────────────
print(f"Estrazione feature spettrali ({FEATURE_PARAMS['n_channels']} canali x {FEATURE_PARAMS['n_bands']} bande) with temporal stacking W={FEATURE_PARAMS['W']}...")
X_feat = extract_features(X)
y_stacked = y[FEATURE_PARAMS['W'] - 1:]  # label dell'ultima finestra del triplet
print(f"   Shape feature stacked: {X_feat.shape}, label: {y_stacked.shape}")

# ─── Train / Test split ───────────────────────────────────────────────────────
split = int(len(y_stacked) * TRAIN_PARAMS['train_split'])
X_train, X_test = X_feat[:split], X_feat[split:]
y_train, y_test = y_stacked[:split], y_stacked[split:]

# ─── Cross-Validation (5-fold) — proxy della loss curve ──────────────────────
print(f"\nCross-Validation ({TRAIN_PARAMS['cv_folds']}-fold stratificato)...")
model_cv = build_model()
cv = StratifiedKFold(n_splits=TRAIN_PARAMS['cv_folds'], shuffle=False) # shuffle=False per mantenere l'ordine temporale, random_state è irrilevante in questo caso
cv_scores = cross_val_score(model_cv, X_train, y_train,
                            cv=cv, scoring=TRAIN_PARAMS['scoring'])
for i, s in enumerate(cv_scores, 1):
    print(f"   Fold {i}: bAcc = {s:.4f}")
print(f"   Media CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Grafico CV scores (loss curve proxy) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
folds = list(range(1, len(cv_scores) + 1))
ax.bar(folds, cv_scores, color='steelblue', alpha=0.8, label='bAcc per fold')
ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Media = {cv_scores.mean():.4f}')
ax.fill_between(folds,
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color='red', label=f'±1 std = {cv_scores.std():.4f}')
ax.set_xlabel('Fold')
ax.set_ylabel('Balanced Accuracy')
ax.set_title(f'SVM Cross-Validation — {PATIENT_ID}')
ax.set_ylim(0, 1.05)
ax.set_xticks(folds)
ax.legend()
plt.tight_layout()
plot_path = PLOTS_DIR / f"{PATIENT_ID}_ml_svm_cv.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"   📈 Grafico CV salvato in: {plot_path}")

# ─── Training finale su tutto il train set ────────────────────────────────────
print("\nTraining SVM (RBF kernel) — fit finale...")
model = build_model()
model.fit(X_train, y_train)
print("   Training completato ✅")

# ─── Valutazione ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
bAcc = balanced_accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n✅ Risultati {PATIENT_ID} — Classic ML (SVM):")
print(f"   Balanced Accuracy : {bAcc:.4f}")
print(f"   Sensitivity       : {sensitivity:.4f}")
print(f"   Specificity       : {specificity:.4f}")
print(f"   TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"   CV Media          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Salva parametri JSON ─────────────────────────────────────────────────────
params_log = {
    'patient':       PATIENT_ID,
    'model':         'Classic_ML_SVM',
    'svm_params':    SVM_PARAMS,
    'feature_params': FEATURE_PARAMS,
    'train_params':  TRAIN_PARAMS,
    'n_features': int(X_feat.shape[1]),  # 504 = 21 canali × 8 bande × W=3
    'n_train':       int(len(X_train)),
    'n_test':        int(len(X_test)),
    'n_ictal_train': int(y_train.sum()),
    'n_ictal_test':  int(y_test.sum()),
    'cv_scores':     [round(float(s), 4) for s in cv_scores],
    'cv_mean':       round(float(cv_scores.mean()), 4),
    'cv_std':        round(float(cv_scores.std()), 4),
    'test_bAcc':     round(float(bAcc), 4),
    'sensitivity':   round(float(sensitivity), 4),
    'specificity':   round(float(specificity), 4),
}
params_path = PARAMS_DIR / f"{PATIENT_ID}_ml_svm_params.json"
with open(params_path, 'w') as f:
    json.dump(params_log, f, indent=2)
print(f"   🔧 Parametri salvati in: {params_path}")

# ─── Salva risultati CSV (schema unificato benchmark) ─────────────────────────
csv_path = RESULTS_DIR / "benchmark_results.csv"
write_header = not csv_path.exists()
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model","patient","bAcc","sensitivity","specificity","TP","FP","TN","FN",
        "cv_mean","cv_std","final_loss","stopped_epoch"])
    if write_header:
        writer.writeheader()
    writer.writerow({
        "model": "Classic_ML_SVM", "patient": PATIENT_ID,
        "bAcc": round(bAcc, 4), "sensitivity": round(sensitivity, 4), "specificity": round(specificity, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "cv_mean":    round(float(cv_scores.mean()), 4),
        "cv_std":     round(float(cv_scores.std()),  4),
        "final_loss": "",   # non applicabile per SVM
        "stopped_epoch": "N/A",
    })
print(f"  📊 Risultati salvati in benchmark_results.csv")

# ─── Salva modello ────────────────────────────────────────────────────────────
ckpt_path = CHECKPOINT_DIR / f"{PATIENT_ID}_ml_svm.pkl"
joblib.dump(model, ckpt_path)
print(f"   💾 Modello salvato in: {ckpt_path}")
