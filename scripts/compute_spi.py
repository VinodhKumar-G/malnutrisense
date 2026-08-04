"""
scripts/compute_spi.py — Compute SPI-3 and merge satellite features onto NFHS-5.
 
Reads:  data/raw/external/ndvi_districts.csv
        data/raw/external/chirps_monthly_districts.csv
        data/processed/nfhs5_cleaned.csv
 
Writes: data/processed/nfhs5_with_satellite.csv
        reports/tables/satellite_coverage.csv
"""
 
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import NFHS5_CLEANED_PATH, PROCESSED_DIR, TABLES_DIR
from src.logger import get_console_logger, CleaningLogger
 
log  = get_console_logger(__name__)
clog = CleaningLogger()


def normalize_text(value):
    if pd.isna(value):
        return ''
    return re.sub(r'[^a-z0-9]+', '', str(value).strip().lower())


def find_column(df, candidates):
    normalized_columns = {normalize_text(col): col for col in df.columns}
    for candidate in candidates:
        if normalize_text(candidate) in normalized_columns:
            return normalized_columns[normalize_text(candidate)]
    return None


def resolve_state_names(df, state_col, state_code_map):
    if state_col is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    resolved = []
    for value in df[state_col]:
        if pd.isna(value):
            resolved.append(np.nan)
            continue

        try:
            code = int(float(value))
        except (TypeError, ValueError):
            code = None

        if code is not None and code in state_code_map:
            resolved.append(state_code_map[code])
            continue

        text = str(value).strip()
        if not text:
            resolved.append(np.nan)
            continue

        normalized_text = normalize_text(text)
        matched = None
        for name in state_code_map.values():
            if normalize_text(name) == normalized_text:
                matched = name
                break
        resolved.append(matched if matched is not None else text)

    return pd.Series(resolved, index=df.index)


def compute_spi_3(monthly_df):
    '''
    Compute SPI-3 (3-month SPI) for each district.
    SPI-3 measures rainfall anomaly over a 3-month window.
    Positive = wetter than normal, Negative = drier (drought).
    '''
    spi_results = []
    for district, group in monthly_df.groupby(['ADM1_NAME','ADM2_NAME']):
        group = group.sort_values(['year','month'])
        # 3-month rolling sum
        group['rain_3month'] = group['mean'].rolling(3, min_periods=3).sum()
        # SPI = (x - mean) / std, fitted to gamma distribution
        rain_vals = group['rain_3month'].dropna()
        if len(rain_vals) < 6:
            continue
        # Fit normal distribution and compute z-scores
        mu, sigma = rain_vals.mean(), rain_vals.std()
        if sigma > 0:
            group['spi_3'] = (group['rain_3month'] - mu) / sigma
        else:
            group['spi_3'] = 0.0
        # Keep only 2019-2021 period mean SPI
        survey_period = group[group['year'].between(2019, 2021)]
        spi_mean = survey_period['spi_3'].mean()
        spi_results.append({
            'ADM1_NAME': district[0],
            'ADM2_NAME': district[1],
            'spi_3_mean_2019_2021': round(spi_mean, 4),
        })
    return pd.DataFrame(spi_results)
 
def main():
    # Load satellite CSVs
    ndvi_path  = Path('data/raw/external/ndvi_districts.csv')
    chirps_path = Path('data/raw/external/chirps_monthly_districts.csv')
 
    if not ndvi_path.exists() or not chirps_path.exists():
        print('ERROR: Download satellite CSVs from Google Drive first.')
        print(f'  NDVI path:   {ndvi_path}')
        print(f'  CHIRPS path: {chirps_path}')
        return
 
    ndvi   = pd.read_csv(ndvi_path)
    chirps = pd.read_csv(chirps_path)
 
    # Compute SPI-3
    log.info('Computing SPI-3...')
    spi_df = compute_spi_3(chirps)
    log.info(f'SPI computed for {len(spi_df)} districts')
 
    # Merge NDVI + SPI into one satellite feature table
    ndvi_renamed = ndvi.rename(columns={'mean': 'mean_ndvi_2019_2021'})
    satellite = ndvi_renamed.merge(
        spi_df, on=['ADM1_NAME','ADM2_NAME'], how='outer'
    )
    satellite.columns = [c.lower().replace(' ','_') for c in satellite.columns]
 
    # Load NFHS-5 cleaned dataset
    df = pd.read_csv(NFHS5_CLEANED_PATH)
    log.info(f'NFHS-5 loaded: {len(df):,} rows')
 
    # Merge satellite features onto NFHS-5 using state-level mapping
    state_code_map = {
        1: 'Jammu And Kashmir',
        2: 'Himachal Pradesh',
        3: 'Punjab',
        4: 'Chandigarh',
        5: 'Uttarakhand',
        6: 'Haryana',
        7: 'Delhi',
        8: 'Rajasthan',
        9: 'Uttar Pradesh',
        10: 'Bihar',
        11: 'Sikkim',
        12: 'Arunachal Pradesh',
        13: 'Nagaland',
        14: 'Manipur',
        15: 'Mizoram',
        16: 'Tripura',
        17: 'Meghalaya',
        18: 'Assam',
        19: 'West Bengal',
        20: 'Jharkhand',
        21: 'Odisha',
        22: 'Chhattisgarh',
        23: 'Madhya Pradesh',
        24: 'Gujarat',
        25: 'Daman and Diu',
        27: 'Maharashtra',
        28: 'Andhra Pradesh',
        29: 'Karnataka',
        30: 'Goa',
        31: 'Lakshadweep',
        32: 'Kerala',
        33: 'Tamil Nadu',
        34: 'Puducherry',
        35: 'Andaman and Nicobar Islands',
        36: 'Telangana',
        37: 'Ladakh',
    }

    state_col = find_column(df, ['v024', 'state_code', 'state', 'state_name', 'stateid'])
    if state_col is None:
        raise KeyError(f'Could not find a state column in NFHS data. Available columns: {list(df.columns)}')

    df['state_name'] = resolve_state_names(df, state_col, state_code_map)
    df['state_name_norm'] = df['state_name'].apply(normalize_text)

    # Aggregate satellite to state level (fallback if district GPS not available)
    sat_state = satellite.copy()
    sat_state['adm1_name_norm'] = sat_state['adm1_name'].fillna('').apply(normalize_text)
    sat_state = sat_state.groupby('adm1_name_norm').agg(
        mean_ndvi=('mean_ndvi_2019_2021','mean'),
        spi_3=('spi_3_mean_2019_2021','mean'),
    ).reset_index()
    sat_state = sat_state.rename(columns={'adm1_name_norm': 'state_name_norm'})

    df = df.merge(sat_state, on='state_name_norm', how='left')

    # Coverage check
    coverage = df['mean_ndvi'].notna().mean()
    log.info(f'Satellite coverage: {coverage:.1%} of records')

    coverage_table = (
        df.groupby('state_name', dropna=False)
        .agg(records=('state_name', 'size'), with_satellite=('mean_ndvi', lambda s: int(s.notna().sum())), coverage_pct=('mean_ndvi', lambda s: s.notna().mean()))
        .reset_index()
    )
    coverage_table['coverage_pct'] = coverage_table['coverage_pct'] * 100

    # Save enriched dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PROCESSED_DIR / 'nfhs5_with_satellite.csv'
    df.to_csv(out_path, index=False)

    coverage_path = TABLES_DIR / 'satellite_coverage.csv'
    coverage_table.to_csv(coverage_path, index=False)

    log.info(f'Saved: {out_path} ({len(df):,} rows, {df.shape[1]} cols)')
    log.info(f'Saved coverage report: {coverage_path}')
    print(f'Satellite features merged. Coverage: {coverage:.1%}')
    print(f'Saved: {out_path}')
    print(f'Saved: {coverage_path}')
 
if __name__ == '__main__':
    main()
