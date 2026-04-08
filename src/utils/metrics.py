"""
═══════════════════════════════════════════════════════════════════════════════
                      METRICS - Evaluation Metrics Computation
═══════════════════════════════════════════════════════════════════════════════

This module computes evaluation metrics for the seizure detection model, including:
  - Balanced Accuracy (bAcc): geometric mean of sensitivity and specificity
  - Sensitivity: percentage of correctly detected seizures
  - False Positives per hour (FP/h): number of false alarms per recording hour
  - Confusion Matrix: TP, FP, TN, FN
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def compute_metrics(y_true, y_pred, recording_hours=None):
    bAcc = balanced_accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    fp_per_hour = (fp / recording_hours) if recording_hours else None
    
    return {
        "bAcc":        round(bAcc, 4),
        "sensitivity": round(sensitivity, 4),
        "FP/h":        round(fp_per_hour, 2) if fp_per_hour else "N/A",
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }
