"""
scripts/compute_district_stats.py — Aggregate NFHS-5 records to district level.
 
Creates reports/tables/district_prevalence.csv with one row per district.
Usage: python3 scripts/compute_district_stats.py
"""
 
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import NFHS5_CLEANED_PATH, NFHS5_PATH, TABLES_DIR, TARGET_COLS
from src.logger import get_console_logger
from src.utils import format_number
 
log = get_console_logger(__name__)
 
def main():
    if NFHS5_PATH.exists():
        log.info('Loading raw NFHS-5 DTA with district identifiers...')
        df, _ = pyreadstat.read_dta(
            str(NFHS5_PATH),
            usecols=['v024', 'sdist', 'v001', 'hw70', 'hw71', 'hw72'],
            encoding='utf-8',
        )
        df = df.replace([9996, 9997, 9998, 9999, 99996, 99997, 99998, 99999], np.nan)

        for raw_col, new_col in [('hw70', 'HAZ'), ('hw71', 'WAZ'), ('hw72', 'WHZ')]:
            if raw_col in df.columns:
                df[new_col] = pd.to_numeric(df[raw_col], errors='coerce') / 100.0
                df.drop(columns=[raw_col], inplace=True)

        df['stunted'] = (df['HAZ'] < -2).astype(int)
        df['underweight'] = (df['WAZ'] < -2).astype(int)
        df['wasted'] = (df['WHZ'] < -2).astype(int)
    else:
        log.info('Loading cleaned NFHS-5 dataset...')
        df = pd.read_csv(NFHS5_CLEANED_PATH)

    log.info(f'Loaded: {format_number(len(df))} rows')

    state_col = next(
        (c for c in df.columns if c.lower() in {'v024', 'state_code', 'state_id', 'state', 'state_name'}),
        None,
    )
    district_col = next(
        (c for c in df.columns if c.lower() in {'sdist', 'shdist', 'district', 'district_name', 'district_id', 'district_code', 'dist_name', 'dist_id'}),
        None,
    )

    if state_col is None:
        raise KeyError(f'Could not find a state identifier column. Available columns: {list(df.columns)}')
    if district_col is None:
        raise KeyError(f'Could not find a district identifier column. Available columns: {list(df.columns)}')

    if state_col != 'V024':
        df = df.rename(columns={state_col: 'V024'})
    if district_col != 'district_code':
        df = df.rename(columns={district_col: 'district_code'})

    district_stats = (
        df.groupby(['V024', 'district_code'])
        .agg(
            stunting_prev    = ('stunted',     'mean'),
            underweight_prev = ('underweight', 'mean'),
            wasting_prev     = ('wasted',      'mean'),
            mean_haz         = ('HAZ',         'mean'),
            mean_waz         = ('WAZ',         'mean'),
            mean_whz         = ('WHZ',         'mean'),
            n_children       = ('stunted',     'count'),
        )
        .reset_index()
    )

    district_stats['is_aspirational'] = 0
 
    # Round percentages
    for col in ['stunting_prev','underweight_prev','wasting_prev']:
        district_stats[col] = district_stats[col].round(4)
 
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / 'district_prevalence.csv'
    district_stats.to_csv(out_path, index=False)
 
    log.info(f'Saved: {out_path}  ({len(district_stats)} districts)')
    print(f'Districts computed: {len(district_stats)}')
    print(f'Mean stunting prevalence: {district_stats["stunting_prev"].mean():.1%}')
    print(f'Saved: {out_path}')
 
if __name__ == '__main__':
    main()
