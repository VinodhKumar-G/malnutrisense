"""
src/federated/partition.py — Partition NFHS-5 into state-level federated nodes.
 
5 node states (highest burden states covering ~60% of malnourished children):
  UP (9), Bihar (10), MP (23), Rajasthan (8), Maharashtra (27)
"""
 
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR, TRAIN_TEST_DIR, TARGET_COLS
from src.logger import get_console_logger
from src.preprocessing import make_train_test_split
 
log = get_console_logger(__name__)
 
# 5 highest-burden states — V024 codes
FL_NODES = {
    'Uttar_Pradesh':   9,
    'Bihar':           10,
    'Madhya_Pradesh':  23,
    'Rajasthan':       8,
    'Maharashtra':     27,
}
 
def create_partitions(df: pd.DataFrame) -> dict:
    """
    Split NFHS-5 data into 5 state partitions.
    Returns dict: {state_name: (X_train, X_test, y_train, y_test)}
    """
    partitions = {}
    for state_name, state_code in FL_NODES.items():
        state_df = df[df['V024'] == state_code].copy()
        log.info(f'{state_name}: {len(state_df):,} rows')
 
        if len(state_df) < 1000:
            log.warning(f'{state_name} has fewer than 1000 rows — skipping')
            continue
 
        X_tr, X_te, y_tr, y_te = make_train_test_split(state_df)
        partitions[state_name] = (X_tr, X_te, y_tr, y_te)
 
    log.info(f'Created {len(partitions)} state partitions')
    return partitions
