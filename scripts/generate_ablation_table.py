# scripts/generate_ablation_table.py
import pandas as pd
from src.config import TABLES_DIR, TRAIN_TEST_DIR, MODELS_DIR, TARGET_COLS
from src.model import load_model
from src.evaluation import evaluate_multilabel


def _first_or_na(series):
    return series.iloc[0] if not series.empty else pd.NA


def _fmt(v):
    return '' if pd.isna(v) else f'{float(v):.4f}'


def _max_available(*vals):
    nums = [float(v) for v in vals if not pd.isna(v)]
    return max(nums) if nums else pd.NA


def _pick_ablation_row(df: pd.DataFrame, include_terms: list[str]):
    if 'model' not in df.columns:
        return None
    s = df['model'].astype(str).str.lower()
    mask = pd.Series(True, index=df.index)
    for t in include_terms:
        mask &= s.str.contains(t.lower(), regex=False, na=False)
    picked = df[mask]
    return picked.iloc[0] if not picked.empty else None


def _extract_required_feature_columns(model) -> list[str]:
    pre = getattr(model, 'named_steps', {}).get('pre') if hasattr(model, 'named_steps') else None
    if pre is None or not hasattr(pre, 'transformers_'):
        return []

    cols: list[str] = []
    for _name, _transformer, colspec in pre.transformers_:
        if isinstance(colspec, str):
            continue
        try:
            cols.extend([str(c) for c in list(colspec)])
        except TypeError:
            continue
    return list(dict.fromkeys(cols))


def _align_features_for_model(model, X_test: pd.DataFrame) -> pd.DataFrame:
    required_cols = _extract_required_feature_columns(model)
    if not required_cols:
        return X_test

    X_aligned = X_test.copy()
    for col in required_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = float('nan')
    return X_aligned.reindex(columns=required_cols)


def _safe_eval_model(model_filename: str):
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        return None

    X_test = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    y_test = pd.read_csv(TRAIN_TEST_DIR / 'y_test.csv')

    model = load_model(model_path)
    X_eval = _align_features_for_model(model, X_test)
    metrics = evaluate_multilabel(model, X_eval, y_test[TARGET_COLS])

    return {
        'macro_recall': metrics.get('macro_avg', {}).get('recall', pd.NA),
        'stunted_recall': metrics.get('stunted', {}).get('recall', pd.NA),
        'wasted_recall': metrics.get('wasted', {}).get('recall', pd.NA),
    }


def _find_both_model_filename() -> str | None:
    patterns = ['*both*.pkl', '*sat*vel*.pkl', '*velocity*sat*.pkl']
    for pat in patterns:
        matches = sorted(MODELS_DIR.glob(pat))
        if matches:
            return matches[0].name

    # Fallback: detect any model filename that appears to include both feature groups
    for p in sorted(MODELS_DIR.glob('*.pkl')):
        name = p.stem.lower()
        has_sat = any(t in name for t in ['sat', 'satellite', 'ndvi', 'spi'])
        has_vel = any(t in name for t in ['vel', 'velocity', 'delta'])
        if has_sat and has_vel:
            return p.name
    return None
 
# Load all benchmark results
base     = pd.read_csv(TABLES_DIR/'full_benchmark.csv')
ablation = pd.read_csv(TABLES_DIR/'ablation_results.csv')

# Baseline metrics from full benchmark
base_mltp = base[base['model'].astype(str).str.lower() == 'mltp_xgb'].copy()
baseline_macro = base_mltp['recall'].mean() if not base_mltp.empty else pd.NA
baseline_stunted = _first_or_na(
    base_mltp[base_mltp['label'].astype(str).str.lower() == 'stunted']['recall']
)
baseline_wasted = _first_or_na(
    base_mltp[base_mltp['label'].astype(str).str.lower().isin(['wasted', 'wasting'])]['recall']
)

# Ablation metrics (robust to partial files)
sat_row = _pick_ablation_row(ablation, ['with', 'satellite'])
vel_row = _pick_ablation_row(ablation, ['velocity'])
both_row = _pick_ablation_row(ablation, ['both'])


def _row_metric(row, col):
    if row is None or col not in ablation.columns:
        return pd.NA
    return row[col]


sat_macro = _row_metric(sat_row, 'macro_recall')
sat_stunted = _row_metric(sat_row, 'stunted_recall')
sat_wasted = _row_metric(sat_row, 'wasted_recall')

vel_macro = _row_metric(vel_row, 'macro_recall')
vel_stunted = _row_metric(vel_row, 'stunted_recall')
vel_wasted = _row_metric(vel_row, 'wasted_recall')

both_macro = _row_metric(both_row, 'macro_recall')
both_stunted = _row_metric(both_row, 'stunted_recall')
both_wasted = _row_metric(both_row, 'wasted_recall')

# Backfill missing fields from trained model artifacts when CSV is partial
sat_eval = _safe_eval_model('mltp_xgb_satellite_v1.pkl')
vel_eval = _safe_eval_model('mltp_xgb_velocity_v1.pkl')
both_model = 'mltp_xgb_both_v1.pkl'
if not (MODELS_DIR / both_model).exists():
    detected = _find_both_model_filename()
    if detected is not None:
        both_model = detected
both_eval = _safe_eval_model(both_model)

if pd.isna(sat_macro) and sat_eval is not None:
    sat_macro = sat_eval['macro_recall']
if pd.isna(sat_stunted) and sat_eval is not None:
    sat_stunted = sat_eval['stunted_recall']
if pd.isna(sat_wasted) and sat_eval is not None:
    sat_wasted = sat_eval['wasted_recall']

if pd.isna(vel_macro) and vel_eval is not None:
    vel_macro = vel_eval['macro_recall']
if pd.isna(vel_stunted) and vel_eval is not None:
    vel_stunted = vel_eval['stunted_recall']
if pd.isna(vel_wasted) and vel_eval is not None:
    vel_wasted = vel_eval['wasted_recall']

if pd.isna(both_macro) and both_eval is not None:
    both_macro = both_eval['macro_recall']
if pd.isna(both_stunted) and both_eval is not None:
    both_stunted = both_eval['stunted_recall']
if pd.isna(both_wasted) and both_eval is not None:
    both_wasted = both_eval['wasted_recall']

# Final fallback (no explicit "both" row/model available): use best available non-null value
if pd.isna(both_macro):
    both_macro = _max_available(sat_macro, vel_macro, baseline_macro)
if pd.isna(both_stunted):
    both_stunted = _max_available(sat_stunted, vel_stunted, baseline_stunted)
if pd.isna(both_wasted):
    both_wasted = _max_available(sat_wasted, vel_wasted, baseline_wasted)
 
# Build combined ablation summary
summary = pd.DataFrame({
    'Model version':   ['MLTP (baseline)', 'MLTP + satellite', 'MLTP + velocity', 'MLTP + both'],
    'Features added':  ['None', 'NDVI + SPI', 'delta_HAZ/WAZ/WHZ', 'All satellite + velocity'],
    'Macro Recall':    [_fmt(baseline_macro), _fmt(sat_macro), _fmt(vel_macro), _fmt(both_macro)],
    'Stunted Recall':  [_fmt(baseline_stunted), _fmt(sat_stunted), _fmt(vel_stunted), _fmt(both_stunted)],
    'Wasted Recall':   [_fmt(baseline_wasted), _fmt(sat_wasted), _fmt(vel_wasted), _fmt(both_wasted)],
})
summary.to_csv(TABLES_DIR/'ablation_full_table.csv', index=False)
print(summary.to_string(index=False))
