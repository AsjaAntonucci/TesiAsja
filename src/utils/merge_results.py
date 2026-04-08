"""
═══════════════════════════════════════════════════════════════════════════════
                    MERGE RESULTS - Aggregating Model Metrics
═══════════════════════════════════════════════════════════════════════════════

MERGE STEPS:
  1. DISCOVER FILES    → Scans results/metrics/ for per-patient result files
  2. AGGREGATE ROWS    → Collects all metrics across patients and models
  3. STANDARDIZE       → Ensures consistent column order and formatting
  4. EXPORT BENCHMARK  → Writes consolidated benchmark_results.csv

INPUT:
  results/metrics/chb*_*_results.csv (per-patient result files from training)

OUTPUT:
  results/metrics/benchmark_results.csv (unified benchmark file)

FIELDS:
  model | patient | bAcc | sensitivity | specificity | TP | FP | TN | FN | cv_mean | cv_std | final_loss | stopped_epoch
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
from pathlib import Path

RESULTS_DIR = Path("results") / "metrics"
FIELDNAMES  = ["model","patient","bAcc","sensitivity","specificity",
               "TP","FP","TN","FN","cv_mean","cv_std","final_loss","stopped_epoch"]

all_rows = []
for f in sorted(RESULTS_DIR.glob("chb*_*_results.csv")):
    with open(f, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            all_rows.append(row)

out_path = RESULTS_DIR / "benchmark_results.csv"
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"[OK] Merge completato — {len(all_rows)} righe in {out_path}")