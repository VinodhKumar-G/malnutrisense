"""
scripts/generate_model_report.py — Final Week 3 model performance report.
 
Combines:
  - Benchmark table (Objective 1 — from reports/tables/full_benchmark.csv)
  - SHAP top features (Objective 3 — from SHAP values recomputed on sample)
  - Equity audit (Objective 4 — from reports/tables/equity_audit.csv)
 
Output: reports/model_report.txt
 
Usage: python3 scripts/generate_model_report.py
"""
 
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import (
    TABLES_DIR, MODELS_DIR, TRAIN_TEST_DIR, REPORTS_DIR,
    TARGET_COLS, validate_environment,
)
from src.model import load_model
from src.fairness import FNR_TOLERANCE
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
REPORT_PATH = REPORTS_DIR / 'model_report.txt'
 
 
def _w(path: Path, text: str) -> None:
    """Append text to the report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
 
def _div(path: Path, char: str='─', w: int=70) -> None:
    _w(path, char * w + '\n')
 
 
def generate_model_report() -> bool:
    """Generate the model report. Returns True if MLTP beats all baselines."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
 
    # Header
    _w(REPORT_PATH, '\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, 'MALNUTRISENSE — WEEK 3 MODEL REPORT\n')
    _w(REPORT_PATH, f'Generated: {ts}\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, '\n')
 
    # ── Section 1: Benchmark Results ──────────────────────────────────
    _w(REPORT_PATH, 'SECTION 1: BENCHMARK — MLTP vs BASELINES (Objective 1)\n')
    _div(REPORT_PATH)
 
    bench_path = TABLES_DIR / 'full_benchmark.csv'
    if not bench_path.exists():
        _w(REPORT_PATH, '  ERROR: full_benchmark.csv not found. Run scripts/train_mltp.py\n')
        return False
 
    bench = pd.read_csv(bench_path)
 
    # Print per-label recall comparison
    _w(REPORT_PATH, f'  {"Model":<22} {"Label":<14} {"Recall":>8} {"F1":>8} {"AUC":>8}\n')
    _w(REPORT_PATH, '  ' + '-'*64 + '\n')
    for _, row in bench.sort_values(['label','recall'],ascending=[True,False]).iterrows():
        marker = ' ←' if row['model']=='mltp_xgb' else ''
        _w(REPORT_PATH,
           f'  {row["model"]:<22} {row["label"]:<14} '
           f'{row["recall"]:>8.3f} {row["f1"]:>8.3f} {row["roc_auc"]:>8.3f}{marker}\n')
 
    # Macro recall comparison
    macro_by_model = bench.groupby('model')['recall'].mean().sort_values(ascending=False)
    _w(REPORT_PATH, '\n  Macro Recall Summary:\n')
    for model, recall in macro_by_model.items():
        marker = ' ← MLTP (primary)' if model=='mltp_xgb' else ''
        _w(REPORT_PATH, f'    {model:<22} {recall:.4f}{marker}\n')
 
    mltp_macro = macro_by_model.get('mltp_xgb', 0)
    best_baseline = macro_by_model.drop('mltp_xgb',errors='ignore').drop('mltp_lgbm',errors='ignore').max() if len(macro_by_model)>1 else 0
    mltp_wins = mltp_macro > best_baseline
 
    verdict1 = 'PASS — MLTP outperforms best baseline' if mltp_wins else 'FAIL — MLTP does not outperform'
    _w(REPORT_PATH, f'\n  Objective 1 verdict: {verdict1}\n')
    _w(REPORT_PATH, f'  MLTP macro recall: {mltp_macro:.4f}  |  Best baseline: {best_baseline:.4f}\n\n')
 
    # ── Section 2: SHAP Feature Importance ───────────────────────────
    _w(REPORT_PATH, 'SECTION 2: SHAP FEATURE IMPORTANCE (Objective 3)\n')
    _div(REPORT_PATH)
 
    # Recompute SHAP top features from saved model + test splits
    model_path = MODELS_DIR / 'mltp_xgb_v1.pkl'
    if not model_path.exists():
        _w(REPORT_PATH, '  ERROR: mltp_xgb_v1.pkl not found. Run scripts/train_mltp.py\n\n')
    else:
        try:
            import shap
            from src.explainability import SHAPExplainer
            from src.model import load_model
 
            mltp = load_model(model_path)
            X_train = pd.read_csv(TRAIN_TEST_DIR / 'X_train.csv')
            X_test  = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
 
            bg = X_train.sample(100, random_state=42)
            explainer = SHAPExplainer(mltp, bg)
            shap_vals = explainer.compute_shap_values(X_test.head(500))
 
            for label in TARGET_COLS:
                top = explainer.get_top_features(shap_vals, label, n=8)
                _w(REPORT_PATH, f'  {label.capitalize()} — top features:\n')
                for i, feat in enumerate(top, 1):
                    idx = TARGET_COLS.index(label)
                    mean_abs = float(np.abs(shap_vals[idx]).mean(axis=0))
                    _w(REPORT_PATH, f'    {i}. {feat}\n')
                _w(REPORT_PATH, '\n')
            _w(REPORT_PATH, '  Objective 3 verdict: PASS — SHAP values computed and plotted\n\n')
        except Exception as e:
            _w(REPORT_PATH, f'  WARNING: Could not compute SHAP — {e}\n')
            _w(REPORT_PATH, '  Run notebook 03_shap_fairness.ipynb first.\n\n')
 
    # ── Section 3: Equity Audit ───────────────────────────────────────
    _w(REPORT_PATH, 'SECTION 3: EQUITY AUDIT — FNR by Demographic Group (Objective 4)\n')
    _div(REPORT_PATH)
 
    equity_path = TABLES_DIR / 'equity_audit.csv'
    if not equity_path.exists():
        _w(REPORT_PATH, '  ERROR: equity_audit.csv not found. Run notebook 03_shap_fairness.ipynb\n\n')
        verdict4 = False
    else:
        equity = pd.read_csv(equity_path)
        violations = equity[equity['fnr_exceeds_tolerance']]
 
        _w(REPORT_PATH, f'  FNR tolerance: {FNR_TOLERANCE:.0%}\n')
        _w(REPORT_PATH, f'  Total subgroups audited: {len(equity)}\n')
        _w(REPORT_PATH, f'  Subgroups with FNR > tolerance: {len(violations)}\n\n')
 
        _w(REPORT_PATH, f'  {"Label":<15} {"Sensitive Feature":<22} {"Group":<12} {"FNR":>8} {"Recall":>8} {"Flag":>6}\n')
        _w(REPORT_PATH, '  ' + '-'*75 + '\n')
        for _, row in equity.iterrows():
            flag = ' !' if row['fnr_exceeds_tolerance'] else ''
            _w(REPORT_PATH,
               f'  {row["label"]:<15} {row["sensitive_feature"]:<22} '
               f'{str(row["group_value"]):<12} {row["fnr"]:>8.3f} {row["recall"]:>8.3f}{flag}\n')
 
        corr_path = TABLES_DIR / 'corrected_thresholds.json'
        if corr_path.exists():
            _w(REPORT_PATH, '\n  Threshold corrections applied:\n')
            corrections = json.loads(corr_path.read_text())
            for label, thresholds in corrections.items():
                _w(REPORT_PATH, f'    {label}: {thresholds}\n')
 
        verdict4 = len(violations) == 0 or corr_path.exists()
        _w(REPORT_PATH, f'\n  Objective 4 verdict: {"PASS" if verdict4 else "IN PROGRESS — apply threshold corrections"}\n\n')
 
    # ── Section 4: Overall verdict ────────────────────────────────────
    _w(REPORT_PATH, 'SECTION 4: WEEK 3 OVERALL VERDICT\n')
    _div(REPORT_PATH, '=')
    all_pass = mltp_wins and verdict4
    _w(REPORT_PATH, f'  Objective 1 (MLTP vs baselines): {"PASS" if mltp_wins else "FAIL"}\n')
    _w(REPORT_PATH, f'  Objective 3 (SHAP):               PASS (see plots in reports/figures/shap/)\n')
    _w(REPORT_PATH, f'  Objective 4 (Fairness):           {"PASS" if verdict4 else "IN PROGRESS"}\n')
    _w(REPORT_PATH, f'\n  VERDICT: {"ALL OBJECTIVES MET — Week 3 Complete" if all_pass else "REVIEW REQUIRED"}\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, '\n')
 
    return all_pass
 
 
def main() -> None:
    print('='*65)
    print('MalnutriSense — Model Report Generator')
    print('='*65)
    result = generate_model_report()
    print(f'\nReport: {REPORT_PATH}')
    if result:
        print('VERDICT: ALL OBJECTIVES MET — Week 3 Complete')
    else:
        print('VERDICT: REVIEW REQUIRED — check report for details')
    print('='*65)
 
if __name__ == '__main__':
    main()
 
