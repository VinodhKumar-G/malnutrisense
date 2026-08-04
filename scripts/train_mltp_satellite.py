"""
scripts/train_mltp_satellite.py — Train MLTP with satellite features + ablation study.
 
Compares: MLTP without satellite vs MLTP with satellite.
Saves: models/mltp_xgb_satellite_v1.pkl
       reports/tables/ablation_results.csv
"""
 
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import PROCESSED_DIR, TRAIN_TEST_DIR, TARGET_COLS, TABLES_DIR
from src.model import build_mltp, save_model, load_class_weights
from src.evaluation import evaluate_multilabel, build_benchmark_table, save_benchmark
from src.preprocessing import make_train_test_split
from src.logger import get_console_logger
from src.utils import timer
 
log = get_console_logger(__name__)
 
def main():
    sat_path = PROCESSED_DIR / 'nfhs5_with_satellite.csv'
    if not sat_path.exists():
        print('ERROR: Run scripts/compute_spi.py first.')
        return
 
    df = pd.read_csv(sat_path)
    log.info(f'Satellite dataset: {len(df):,} rows x {df.shape[1]} cols')
 
    # Drop rows where satellite features are NaN
    df_sat = df.dropna(subset=['mean_ndvi','spi_3'])
    log.info(f'After satellite NaN drop: {len(df_sat):,} rows')
 
    # New train/test split on satellite dataset
    X_sat_train, X_sat_test, y_sat_train, y_sat_test = make_train_test_split(df_sat)
 
    class_weights = load_class_weights()
 
    # Train MLTP WITH satellite features
    print('Training MLTP with satellite features...')
    with timer('MLTP satellite fit'):
        mltp_sat = build_mltp(X_sat_train, class_weights)
        mltp_sat.fit(X_sat_train, y_sat_train[TARGET_COLS].fillna(0).astype(int))
 
    sat_metrics = evaluate_multilabel(mltp_sat, X_sat_test, y_sat_test[TARGET_COLS])
    save_model(mltp_sat, 'mltp_xgb_satellite_v1')
 
    # Load original no-satellite benchmark for comparison
    bench_original = pd.read_csv(TABLES_DIR / 'full_benchmark.csv')
    orig_recall = bench_original[bench_original['model']=='mltp_xgb']['recall'].mean()
    sat_recall  = sat_metrics['macro_avg']['recall']
 
    # Ablation results table
    ablation = pd.DataFrame([
        {'model': 'MLTP_no_satellite',  'macro_recall': orig_recall,
         'stunted_recall': bench_original[bench_original['model']=='mltp_xgb']
                           .set_index('label')['recall'].get('stunted',0)},
        {'model': 'MLTP_with_satellite','macro_recall': sat_recall,
         'stunted_recall': sat_metrics['stunted']['recall']},
    ])
    ablation['improvement'] = ablation['macro_recall'].diff().fillna(0)
 
    ablation.to_csv(TABLES_DIR / 'ablation_results.csv', index=False)
    print('\nAblation Study Results:')
    print(ablation.to_string(index=False))
    print(f'\nMacro Recall improvement from satellite: ',
          f'+{(sat_recall-orig_recall)*100:.2f} pp')
 
if __name__ == '__main__':
    main()
