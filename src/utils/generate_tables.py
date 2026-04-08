"""
═══════════════════════════════════════════════════════════════════════════════
                         TABLE GENERATION - Benchmark Visualization
═══════════════════════════════════════════════════════════════════════════════

GENERATION STEPS:
  1. LOAD DATA         → Reads benchmark results from CSV
  2. MODEL TABLES      → Creates per-model performance tables (CNN, SVM, SPDNet)
  3. COMPARATIVE TABLE → Generates final benchmark comparison across all models
  4. EXPORT PNG        → Saves formatted tables as high-quality PNG files

OUTPUT:
  results/tables/table_CNN.png
  results/tables/table_SVM.png
  results/tables/table_SPDNet.png
  results/tables/table_benchmark_finale.png

USAGE:
  python generate_tables.py
  python generate_tables.py --csv results/metrics/benchmark_results.csv
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Argomenti ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str,
                    default='results/metrics/benchmark_results.csv',
                    help='Percorso al CSV del benchmark')
args = parser.parse_args()

CSV_PATH   = Path(args.csv)
OUTPUT_DIR = Path("results") / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Palette colori ───────────────────────────────────────────────────────────
COLORS = {
    'EpiDeNet_CNN':   '#E07B39',
    'Classic_ML_SVM': '#3A7DC9',
    'SPDNet_classic': '#6BAE75',
}
HEADER_BG = '#1C2E4A'
FOOTER_BG = '#EAF0F8'
ROW_ODD   = '#F7F9FC'
ROW_EVEN  = '#FFFFFF'

# ─── Carica dati ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['patient'] = df['patient'].str.upper()

COLS_MODEL   = ['patient', 'bAcc', 'sensitivity', 'specificity', 'TP', 'FP', 'TN', 'FN', 'cv_mean']
HEADERS_MODEL = ['Paziente', 'bAcc', 'Sensitivity', 'Specificity', 'TP', 'FP', 'TN', 'FN', 'CV Mean']
HEADERS_FINAL = ['Paziente', 'bAcc CNN', 'bAcc SVM', 'bAcc SPDnet', 'Sens CNN', 'Sens SVM', 'Sens SPDnet', 'Best']


# ─── Colori celle ─────────────────────────────────────────────────────────────
def bacc_color(val):
    try:
        v = float(val)
    except Exception:
        return '#FFFFFF'
    if v >= 0.95: return '#C8E6C9'
    elif v >= 0.90: return '#DCEDC8'
    elif v >= 0.80: return '#FFF9C4'
    elif v >= 0.60: return '#FFE0B2'
    else: return '#FFCDD2'


def sens_color(val):
    try:
        v = float(val)
    except Exception:
        return '#FFFFFF'
    if v == 0.0: return '#FFCDD2'
    elif v >= 0.90: return '#C8E6C9'
    elif v >= 0.80: return '#DCEDC8'
    else: return '#FFF9C4'


# ─── Tabella per singolo modello ──────────────────────────────────────────────
def make_model_table(model_key, model_label, color, filename):
    sub = df[df['model'] == model_key][COLS_MODEL].copy()
    sub = sub.sort_values('patient').reset_index(drop=True)

    for c in ['bAcc', 'sensitivity', 'specificity', 'cv_mean']:
        sub[c] = sub[c].apply(lambda x: f'{float(x):.4f}' if pd.notna(x) else '—')
    for c in ['TP', 'FP', 'TN', 'FN']:
        sub[c] = sub[c].apply(lambda x: str(int(x)) if pd.notna(x) else '—')

    n_rows = len(sub)
    fig_h  = 0.45 * n_rows + 1.8
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis('off')

    fig.text(0.5, 0.97, f'Risultati per Paziente — {model_label}',
             ha='center', va='top', fontsize=15, fontweight='bold', color=HEADER_BG)

    table_data = [HEADERS_MODEL] + sub.values.tolist()

    tbl = ax.table(cellText=table_data, cellLoc='center',
                   loc='center', bbox=[0, 0, 1, 0.93])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#D0D8E4')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
            cell.set_height(0.07)
        else:
            val_row = sub.iloc[row - 1]
            cell.set_height(0.052)
            if col == 0:
                cell.set_facecolor(FOOTER_BG)
                cell.set_text_props(fontweight='bold', color=HEADER_BG)
            elif col == 1:
                cell.set_facecolor(bacc_color(val_row['bAcc']))
            elif col == 2:
                cell.set_facecolor(sens_color(val_row['sensitivity']))
            elif col == 8:
                cell.set_facecolor(bacc_color(val_row['cv_mean']))
            else:
                cell.set_facecolor(ROW_ODD if row % 2 == 1 else ROW_EVEN)

    try:
        bacc_vals = sub['bAcc'].apply(float)
        sens_vals = sub['sensitivity'].apply(float)
        mean_b = bacc_vals.mean(); std_b = bacc_vals.std()
        mean_s = sens_vals.mean()
    except Exception:
        mean_b = std_b = mean_s = float('nan')

    fig.text(0.02, 0.01,
             f'Media bAcc: {mean_b:.4f} ± {std_b:.4f}   |   '
             f'Media Sensitivity: {mean_s:.4f}   |   N pazienti: {n_rows}',
             ha='left', va='bottom', fontsize=9, color='#555555', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] {out}')


# ─── Tabella finale comparativa ───────────────────────────────────────────────
def make_final_table():
    cnn = df[df['model'] == 'EpiDeNet_CNN'][['patient', 'bAcc', 'sensitivity']].rename(
        columns={'bAcc': 'bAcc_CNN', 'sensitivity': 'sens_CNN'})
    svm = df[df['model'] == 'Classic_ML_SVM'][['patient', 'bAcc', 'sensitivity']].rename(
        columns={'bAcc': 'bAcc_SVM', 'sensitivity': 'sens_SVM'})
    spd = df[df['model'] == 'SPDNet_classic'][['patient', 'bAcc', 'sensitivity']].rename(
        columns={'bAcc': 'bAcc_SPDnet', 'sensitivity': 'sens_SPDnet'})

    m = cnn.merge(svm, on='patient').merge(spd, on='patient')
    m = m.sort_values('patient').reset_index(drop=True)

    def best(row):
        scores = {'CNN': float(row['bAcc_CNN']),
                  'SVM': float(row['bAcc_SVM']),
                  'SPDNet_classic': float(row['bAcc_SPDnet'])}
        return max(scores, key=scores.get)

    m['best_model'] = m.apply(best, axis=1)

    HEADERS_FINAL = ['Paziente', 'bAcc CNN', 'bAcc SVM', 'bAcc SPDnet', 'Sens CNN', 'Sens SVM', 'Sens SPDnet', 'Best']

    n_rows = len(m)
    fig_h  = 0.44 * n_rows + 2.2
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis('off')

    fig.text(0.5, 0.98, 'Benchmark Comparativo — CHB-MIT (21 Pazienti)',
             ha='center', va='top', fontsize=15, fontweight='bold', color=HEADER_BG)

    rows = []
    for _, r in m.iterrows():
        rows.append([
            r['patient'],
            f"{float(r['bAcc_CNN']):.4f}",
            f"{float(r['bAcc_SVM']):.4f}",
            f"{float(r['bAcc_SPDnet']):.4f}",
            f"{float(r['sens_CNN']):.4f}",
            f"{float(r['sens_SVM']):.4f}",
            f"{float(r['sens_SPDnet']):.4f}",
            r['best_model'],
        ])

    table_data = [HEADERS_FINAL] + rows

    tbl = ax.table(cellText=table_data, cellLoc='center',
                   loc='center', bbox=[0, 0.04, 1, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    best_colors = {
        'CNN':       COLORS['EpiDeNet_CNN'],
        'SVM':       COLORS['Classic_ML_SVM'],
        'SPDNet_classic': COLORS['SPDNet_classic'],
    }

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#D0D8E4')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
            cell.set_height(0.07)
        else:
            r_data = m.iloc[row - 1]
            cell.set_height(0.052)
            if col == 0:
                cell.set_facecolor(FOOTER_BG)
                cell.set_text_props(fontweight='bold', color=HEADER_BG)
            elif col == 1:
                cell.set_facecolor(bacc_color(r_data['bAcc_CNN']))
            elif col == 2:
                cell.set_facecolor(bacc_color(r_data['bAcc_SVM']))
            elif col == 3:
                cell.set_facecolor(bacc_color(r_data['bAcc_SPDnet']))
            elif col == 4:
                cell.set_facecolor(sens_color(r_data['sens_CNN']))
            elif col == 5:
                cell.set_facecolor(sens_color(r_data['sens_SVM']))
            elif col == 6:
                cell.set_facecolor(sens_color(r_data['sens_SPDnet']))
            elif col == 7:
                bm = r_data['best_model']
                cell.set_facecolor(best_colors.get(bm, '#FFFFFF'))
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor(ROW_ODD if row % 2 == 1 else ROW_EVEN)

    mean_cnn    = m['bAcc_CNN'].apply(float).mean()
    mean_svm    = m['bAcc_SVM'].apply(float).mean()
    mean_spd = m['bAcc_SPDnet'].apply(float).mean()
    std_cnn     = m['bAcc_CNN'].apply(float).std()
    std_svm     = m['bAcc_SVM'].apply(float).std()
    std_spd  = m['bAcc_SPDnet'].apply(float).std()
    n_cnn    = (m['best_model'] == 'CNN').sum()
    n_svm    = (m['best_model'] == 'SVM').sum()
    n_spd = (m['best_model'] == 'SPDNet_classic').sum()

    fig.text(0.02, 0.015,
             f'Media bAcc → CNN: {mean_cnn:.4f}±{std_cnn:.4f}   '
             f'SVM: {mean_svm:.4f}±{std_svm:.4f}   '
             f'SPDNet: {mean_spd:.4f}±{std_spd:.4f}   |   '
             f'Vittorie → CNN: {n_cnn}   SVM: {n_svm}   SPDNet: {n_spd}',
             ha='left', va='bottom', fontsize=8.5, color='#555555', style='italic')

    patches = [
        mpatches.Patch(color=COLORS['EpiDeNet_CNN'],   label='Best = CNN'),
        mpatches.Patch(color=COLORS['Classic_ML_SVM'], label='Best = SVM'),
        mpatches.Patch(color=COLORS['SPDNet_classic'], label='Best = SPDNet'),
    ]
    fig.legend(handles=patches, loc='lower right', ncol=3, fontsize=8.5,
               title='Colore colonna Best', title_fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = OUTPUT_DIR / 'table_benchmark_finale.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] {out}')


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nGenerazione tabelle benchmark...\n')
    make_model_table('EpiDeNet_CNN',   'EpiDeNet (CNN)',    COLORS['EpiDeNet_CNN'],   'table_CNN.png')
    make_model_table('Classic_ML_SVM', 'Classic ML (SVM)', COLORS['Classic_ML_SVM'], 'table_SVM.png')
    make_model_table('SPDNet_classic', 'SPDNet',            COLORS['SPDNet_classic'], 'table_SPDNet.png')
    make_final_table()
    print(f'\nTabelle salvate in: {OUTPUT_DIR.resolve()}')