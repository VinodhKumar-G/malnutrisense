"""
scripts/train_mltp_velocity.py — Re-train MLTP with velocity features.
 
Adds delta_haz_per_year, delta_waz_per_year, delta_whz_per_year as features.
Saves: models/mltp_xgb_velocity_v1.pkl
"""
 
import sys
from pathlib import Path
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROCESSED_DIR, TARGET_COLS, TABLES_DIR
from src.model import build_mltp, save_model, load_class_weights
from src.evaluation import evaluate_multilabel, save_benchmark
from src.preprocessing import make_train_test_split
from src.logger import get_console_logger
from src.utils import timer
 
log = get_console_logger(__name__)
 
def main():
    vel_path = PROCESSED_DIR / 'nfhs5_with_velocity.csv'
    if not vel_path.exists():
        print('ERROR: Run scripts/compute_velocity_features.py first.')
        return
 
    df = pd.read_csv(vel_path)
    df = df.dropna(subset=['delta_haz_per_year'])
    log.info(f'Velocity dataset: {len(df):,} rows')
 
    X_train, X_test, y_train, y_test = make_train_test_split(df)
    class_weights = load_class_weights()
 
    with timer('MLTP velocity fit'):
        mltp_vel = build_mltp(X_train, class_weights)
        mltp_vel.fit(X_train, y_train[TARGET_COLS].fillna(0).astype(int))
 
    vel_metrics = evaluate_multilabel(mltp_vel, X_test, y_test[TARGET_COLS])
    save_model(mltp_vel, 'mltp_xgb_velocity_v1')
 
    print(f'Velocity MLTP macro recall: {vel_metrics["macro_avg"]["recall"]:.4f}')
    print('Model saved: models/mltp_xgb_velocity_v1.pkl')
 
if __name__ == '__main__':
    main()
