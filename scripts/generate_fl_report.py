"""
scripts/generate_fl_report.py — Phase 4 federated learning report.
 
Sections:
  1. FL Architecture — 5 state nodes, FedAvg/FedProx, 10 rounds
  2. Convergence — round-by-round macro recall
  3. Privacy-Utility Tradeoff — centralised vs federated comparison
  4. FL Fairness — per-state FNR before and after federated training
 
Reads:  reports/tables/fl_convergence.csv
        reports/tables/fl_vs_centralised.csv
        reports/tables/fl_fairness_comparison.csv
 
Writes: reports/fl_report.txt
 
Usage: python3 scripts/generate_fl_report.py
"""
 
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import TABLES_DIR, REPORTS_DIR
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
REPORT_PATH = REPORTS_DIR / 'fl_report.txt'
 
 
def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
 
def _div(path, char='─', w=70):
    _w(path, char * w + '\n')
 
 
def main():
    # Clear previous report
    REPORT_PATH.unlink(missing_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
 
    _w(REPORT_PATH, '\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, 'MALNUTRISENSE — FEDERATED LEARNING REPORT (Phase 4)\n')
    _w(REPORT_PATH, f'Generated: {ts}\n')
    _div(REPORT_PATH, '=')
 
    # Section 1: Architecture
    _w(REPORT_PATH, '\nSECTION 1: FEDERATED LEARNING ARCHITECTURE\n')
    _div(REPORT_PATH)
    _w(REPORT_PATH, '  Framework:      Flower (flwr)\n')
    _w(REPORT_PATH, '  Strategy:       FedAvg + FedProx (mu=0.1)\n')
    _w(REPORT_PATH, '  Nodes:          5 state nodes (UP, Bihar, MP, Rajasthan, Maharashtra)\n')
    _w(REPORT_PATH, '  Rounds:         10\n')
    _w(REPORT_PATH, '  Base model:     XGBoost MultiOutputClassifier (same as centralised MLTP)\n')
    _w(REPORT_PATH, '  Privacy claim:  Raw data never leaves state boundary\n')
 
    # Section 2: Convergence
    _w(REPORT_PATH, '\nSECTION 2: CONVERGENCE\n')
    _div(REPORT_PATH)
    conv_path = TABLES_DIR / 'fl_convergence.csv'
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
        for _, row in conv.iterrows():
            _w(REPORT_PATH, f'  Round {int(row["round"]):>2}: macro_recall = {row.get("macro_recall","N/A")}\n')
    else:
        _w(REPORT_PATH, '  ERROR: fl_convergence.csv not found\n')
 
    # Section 3: Privacy-Utility Tradeoff
    _w(REPORT_PATH, '\nSECTION 3: PRIVACY-UTILITY TRADEOFF\n')
    _div(REPORT_PATH)
    comp_path = TABLES_DIR / 'fl_vs_centralised.csv'
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        for _, row in comp.iterrows():
            _w(REPORT_PATH, f'  {row["model"]:<22} macro_recall={row["macro_recall"]:.4f}  '
               f'privacy={row.get("privacy","")}\n')
        gap = abs(comp['macro_recall'].diff().iloc[-1])
        _w(REPORT_PATH, f'\n  Recall gap: {gap:.4f}\n')
        _w(REPORT_PATH, f'  Verdict: {"PASS" if gap<0.03 else "REVIEW"} '
           f'(threshold: 3% maximum acceptable gap)\n')
    else:
        _w(REPORT_PATH, '  ERROR: fl_vs_centralised.csv not found\n')
 
    # Section 4: Fairness
    _w(REPORT_PATH, '\nSECTION 4: FEDERATED MODEL FAIRNESS\n')
    _div(REPORT_PATH)
    fair_path = TABLES_DIR / 'fl_fairness_comparison.csv'
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        for model in ['centralised', 'federated']:
            violations = fair[(fair['model']==model) & (fair['fnr_exceeds_tolerance']==True)]
            _w(REPORT_PATH, f'  {model}: {len(violations)} subgroups with FNR > 15%\n')
    else:
        _w(REPORT_PATH, '  ERROR: fl_fairness_comparison.csv not found\n')
 
    _w(REPORT_PATH, '\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, 'END OF FEDERATED LEARNING REPORT\n')
    _div(REPORT_PATH, '=')
 
    print(f'FL report saved: {REPORT_PATH}')
 
if __name__ == '__main__':
    main()

