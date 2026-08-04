"""
scripts/compute_velocity_features.py — Compute Z-score velocity from NFHS-4 to NFHS-5.
 
delta_zscore = (NFHS5_district_mean_HAZ - NFHS4_district_mean_HAZ) / 4.0 years
 
Reads:  data/raw/nfhs4/IAKR74FL.DTA
        reports/tables/district_prevalence.csv  (NFHS-5 district stats)
 
Writes: reports/tables/district_velocity.csv
        data/processed/nfhs5_with_velocity.csv
"""
 
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyreadstat
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import NFHS4_PATH, NFHS5_CLEANED_PATH, MISSING_CODES, TABLES_DIR, PROCESSED_DIR
from src.logger import get_console_logger, CleaningLogger
from src.utils import timer
 
log  = get_console_logger(__name__)
clog = CleaningLogger()
 
YEARS_BETWEEN = 4.0  # NFHS-4: 2015-16, NFHS-5: 2019-21 → ~4 years


def normalize_column_name(col):
    return re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower()).strip('_')


def normalize_columns(df):
    return df.rename(columns=lambda c: normalize_column_name(c))


def resolve_column(df, candidates):
    normalized_map = {normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None


def resolve_required_columns(df, mapping, context):
    resolved = {}
    for key, candidates in mapping.items():
        col = resolve_column(df, candidates)
        if col is None:
            raise KeyError(
                f'{context}: could not find {key} from candidates {candidates}; '
                f'available columns: {list(df.columns)}'
            )
        resolved[key] = col
    return resolved


def main():
    if not NFHS4_PATH.exists():
        print(f'ERROR: NFHS-4 file not found: {NFHS4_PATH}')
        print('Download from dhsprogram.com → India → NFHS-4 → KR file')
        return
 
    # Step 1: Load NFHS-4
    log.info('Loading NFHS-4 KR file...')
    with timer('NFHS-4 load'):
        df4, _ = pyreadstat.read_dta(str(NFHS4_PATH), usecols=['hw70','hw71','hw72','v024'])
    df4 = normalize_columns(df4)
    log.info(f'NFHS-4 loaded: {len(df4):,} rows')

    # Step 2: Resolve required columns robustly
    nfhs4_req = {
        'haz': ['hw70', 'haz4'],
        'waz': ['hw71', 'waz4'],
        'whz': ['hw72', 'whz4'],
        'state': ['v024', 'state', 'state_code', 'stateid'],
    }
    nfhs4_cols = resolve_required_columns(df4, nfhs4_req, 'NFHS-4')

    df4 = df4[[nfhs4_cols['haz'], nfhs4_cols['waz'], nfhs4_cols['whz'], nfhs4_cols['state']]].copy()
    df4 = df4.rename(columns={
        nfhs4_cols['haz']: 'HAZ4',
        nfhs4_cols['waz']: 'WAZ4',
        nfhs4_cols['whz']: 'WHZ4',
        nfhs4_cols['state']: 'V024',
    })

    # Step 3: Replace DHS missing codes
    for col in df4.select_dtypes(include='number').columns:
        df4[col] = df4[col].replace(MISSING_CODES, np.nan)

    # Step 4: Scale Z-scores
    for col in ['HAZ4', 'WAZ4', 'WHZ4']:
        if col in df4.columns:
            df4[col] = df4[col] / 100.0

    # Step 5: Aggregate NFHS-4 to district level
    dist4 = (
        df4.groupby(['V024'])
        .agg(mean_haz4=('HAZ4','mean'), mean_waz4=('WAZ4','mean'), mean_whz4=('WHZ4','mean'))
        .reset_index()
    )

    # Step 6: Load NFHS-5 district stats
    dist5 = pd.read_csv(TABLES_DIR / 'district_prevalence.csv')
    dist5 = normalize_columns(dist5)

    dist5_req = {
        'state': ['v024', 'state', 'state_code', 'stateid'],
    }
    dist5_cols = resolve_required_columns(dist5, dist5_req, 'district_prevalence')
    dist5 = dist5.rename(columns={
        dist5_cols['state']: 'V024',
    })

    # Step 7: Merge on V024
    merged = dist5.merge(dist4, on=['V024'], how='left')
 
    # Step 7: Compute velocity
    # Positive velocity = improvement (HAZ moving toward 0 from negative)
    merged['delta_haz_per_year'] = (
        (merged['mean_haz'] - merged['mean_haz4']) / YEARS_BETWEEN
    ).round(4)
    merged['delta_waz_per_year'] = (
        (merged['mean_waz'] - merged['mean_waz4']) / YEARS_BETWEEN
    ).round(4)
    merged['delta_whz_per_year'] = (
        (merged['mean_whz'] - merged['mean_whz4']) / YEARS_BETWEEN
    ).round(4)
 
    # Save velocity table
    velocity = (
        merged.groupby('V024', as_index=False)
        .agg(
            delta_haz_per_year=('delta_haz_per_year', 'mean'),
            delta_waz_per_year=('delta_waz_per_year', 'mean'),
            delta_whz_per_year=('delta_whz_per_year', 'mean'),
        )
        .dropna(subset=['delta_haz_per_year', 'delta_waz_per_year', 'delta_whz_per_year'])
    )
    velocity.to_csv(TABLES_DIR / 'district_velocity.csv', index=False)
    log.info(f'Velocity computed for {len(velocity)} states')
 
    # Step 8: Merge velocity onto NFHS-5 individual records
    df5 = pd.read_csv(NFHS5_CLEANED_PATH)
    df5 = normalize_columns(df5)

    df5_req = {
        'state': ['v024', 'state', 'state_code', 'stateid'],
    }
    df5_cols = resolve_required_columns(df5, df5_req, 'NFHS-5 cleaned')
    df5 = df5.rename(columns={
        df5_cols['state']: 'V024',
    })

    df5 = df5.merge(velocity, on=['V024'], how='left')
 
    out_path = PROCESSED_DIR / 'nfhs5_with_velocity.csv'
    df5.to_csv(out_path, index=False)
 
    coverage = df5['delta_haz_per_year'].notna().mean()
    log.info(f'Velocity coverage: {coverage:.1%}')
    print(f'Velocity features added. Coverage: {coverage:.1%}')
    print(f'Mean HAZ improvement per year: {velocity["delta_haz_per_year"].mean():.4f}')
    print(f'Saved: {out_path}')
 
if __name__ == '__main__':
    main()
 
