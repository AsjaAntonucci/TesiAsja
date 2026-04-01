"""
═══════════════════════════════════════════════════════════════════════════════
                      METRICS - Evaluation Metrics Computation
═══════════════════════════════════════════════════════════════════════════════

Questo modulo calcola le metriche di valutazione per il modello di rilevamento
delle crisi epilettiche, inclusi:
  - Balanced Accuracy (bAcc): media geometrica di sensibilità e specificità
  - Sensitivity: percentuale di crisi rilevate correttamente
  - False Positives per ora (FP/h): numero di falsi allarmi per ora di registrazione
  - Matrice di confusione: TP, FP, TN, FN
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def compute_metrics(y_true, y_pred, recording_hours=None):
    # Calcola la Balanced Accuracy (media tra sensibilità e specificità)
    bAcc           = balanced_accuracy_score(y_true, y_pred)
    
    # Ottiene la matrice di confusione e la converte in scalari (tn, fp, fn, tp)
    # TN = True Negatives (crisi non rilevate correttamente)
    # FP = False Positives (falsi allarmi - no crisi predetti come crisi)
    # FN = False Negatives (crisi non rilevate - crisi predetti come no crisi)
    # TP = True Positives (crisi rilevate correttamente)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calcola la sensitivity (recall): percentuale di crisi effettivamente rilevate
    # Formula: TP / (TP + FN)
    sensitivity    = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calcola i falsi allarmi per ora (indicatore critico in una situazione reale)
    # Formula: FP / ore_totali
    fp_per_hour    = (fp / recording_hours) if recording_hours else None
    
    # Ritorna tutte le metriche in un dizionario
    return {
        "bAcc":        round(bAcc, 4),
        "sensitivity": round(sensitivity, 4),
        "FP/h":        round(fp_per_hour, 2) if fp_per_hour else "N/A",
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }
