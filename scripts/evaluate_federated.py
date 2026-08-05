"""
scripts/evaluate_federated.py — Compare federated vs centralised MLTP.
 
Evaluates the federated global model on the full national test set.
Compares against the centralised MLTP benchmark.
 
Reads:  models/fl_global_model.pkl  (saved by FL simulation)
        reports/tables/full_benchmark.csv  (centralised benchmark)
        data/processed/train_test_splits/X_test.csv
        data/processed/train_test_splits/y_test.csv
 
Writes: reports/tables/fl_vs_centralised.csv
 
Usage: python3 scripts/evaluate_federated.py
"""
 
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import TRAIN_TEST_DIR, TABLES_DIR, MODELS_DIR, TARGET_COLS
from src.model import load_model
from src.evaluation import evaluate_multilabel, save_benchmark
from src.logger import get_console_logger
 
log = get_console_logger(__name__)


def _extract_required_feature_columns(model) -> list[str]:
    """Return feature columns expected by the model preprocessor."""
    pre = getattr(model, 'named_steps', {}).get('pre') if hasattr(model, 'named_steps') else None
    if pre is None or not hasattr(pre, 'transformers_'):
        return []

    cols: list[str] = []
    for _name, _transformer, colspec in pre.transformers_:
        if isinstance(colspec, str):
            # Skip selectors like 'drop'/'passthrough'
            continue
        try:
            cols.extend([str(c) for c in list(colspec)])
        except TypeError:
            continue

    # Preserve order while removing duplicates
    return list(dict.fromkeys(cols))
 
 
def main():
    print('='*65)
    print('MalnutriSense — Federated vs Centralised Comparison')
    print('='*65)
 
    # Load national test set
    X_test = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    y_test = pd.read_csv(TRAIN_TEST_DIR / 'y_test.csv')
    log.info(f'National test set: {len(X_test):,} rows')
 
    # Load federated global model
    fl_model_path = MODELS_DIR / 'fl_global_model.pkl'
    if not fl_model_path.exists():
        print(f'ERROR: {fl_model_path} not found.')
        print('Run scripts/run_federated_simulation.py first.')
        return
 
    fl_model = load_model(fl_model_path)
    log.info('Federated global model loaded')

    # Align X_test schema to model training schema
    required_cols = _extract_required_feature_columns(fl_model)
    if required_cols:
        missing_cols = [c for c in required_cols if c not in X_test.columns]
        if missing_cols:
            log.warning(
                f'X_test missing {len(missing_cols)} model feature columns; '
                f'adding as NaN. Sample: {missing_cols[:10]}'
            )
            for col in missing_cols:
                X_test[col] = np.nan
        X_test = X_test.reindex(columns=required_cols)
 
    # Evaluate federated model on national test set
    fl_metrics = evaluate_multilabel(fl_model, X_test, y_test[TARGET_COLS])
    fl_macro = fl_metrics['macro_avg']['recall']
 
    # Load centralised benchmark
    bench = pd.read_csv(TABLES_DIR / 'full_benchmark.csv')
    central_macro = bench[bench['model']=='mltp_xgb']['recall'].mean()
 
    # Compute privacy-utility gap
    gap = central_macro - fl_macro
    gap_pct = (gap / central_macro) * 100
 
    # Build comparison table
    comparison = pd.DataFrame([
        {'model': 'centralised_mltp',  'macro_recall': round(central_macro, 4),
         'training': 'All 232K records on one machine',
         'privacy':  'No privacy — all data centralised'},
        {'model': 'federated_mltp',    'macro_recall': round(fl_macro, 4),
         'training': '5 state nodes, FedAvg, 10 rounds',
         'privacy':  'Data never leaves state boundary'},
    ])
    comparison['recall_gap'] = comparison['macro_recall'].diff().fillna(0).round(4)
    comparison['gap_pct']    = (comparison['recall_gap'] / central_macro * 100).round(2)
 
    # Save
    out_path = TABLES_DIR / 'fl_vs_centralised.csv'
    comparison.to_csv(out_path, index=False)
 
    # Per-label comparison
    print('\nPer-label comparison:')
    print(f'{"Label":<15} {"Centralised":>12} {"Federated":>12} {"Gap":>8}')
    print('-'*50)
    for label in TARGET_COLS:
        c_recall = bench[(bench['model']=='mltp_xgb') & (bench['label']==label)]['recall'].values
        c_val = float(c_recall[0]) if len(c_recall)>0 else 0
        f_val = fl_metrics.get(label, {}).get('recall', 0)
        print(f'{label:<15} {c_val:>12.4f} {f_val:>12.4f} {c_val-f_val:>8.4f}')
 
    print(f'\nMacro Recall: centralised={central_macro:.4f}, federated={fl_macro:.4f}')
    print(f'Privacy-utility gap: {gap:.4f} ({gap_pct:.2f}%)')
 
    if gap_pct <= 3.0:
        print('[PASS] Federated within 3% of centralised — acceptable tradeoff')
    else:
        print(f'[WARN] Federated gap is {gap_pct:.2f}% — consider FedProx (Step 43)')
 
    print(f'\nSaved: {out_path}')
 
if __name__ == '__main__':
    main()
