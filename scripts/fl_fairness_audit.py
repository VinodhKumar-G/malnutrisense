"""
scripts/fl_fairness_audit.py — Compare fairness of federated vs centralised model.
 
Reads:  models/fl_global_model.pkl
        models/mltp_xgb_v1.pkl
        data/processed/nfhs5_with_scst.csv  (or nfhs5_cleaned.csv)
 
Writes: reports/tables/fl_fairness_comparison.csv
        reports/figures/fl_fnr_comparison.png
 
Usage: python3 scripts/fl_fairness_audit.py
"""
 
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from src.config import (
    TRAIN_TEST_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR,
    NFHS5_CLEANED_PATH, TARGET_COLS, PROCESSED_DIR,
)
from src.model import load_model
from src.fairness import FairnessAuditor, FNR_TOLERANCE
from src.logger import get_console_logger
 
log = get_console_logger(__name__)


def _normalize_column_name(col: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(col).strip().lower()).strip('_')


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_map = {_normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None


def _extract_required_feature_columns(model) -> list[str]:
    """Return feature columns expected by model preprocessor."""
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
    """Align test matrix to the model's expected schema (case-insensitive)."""
    required_cols = _extract_required_feature_columns(model)
    if not required_cols:
        return X_test

    normalized_map = {_normalize_column_name(c): c for c in X_test.columns}
    X_aligned = pd.DataFrame(index=X_test.index)

    missing: list[str] = []
    for col in required_cols:
        src = normalized_map.get(_normalize_column_name(col))
        if src is None:
            X_aligned[col] = np.nan
            missing.append(col)
        else:
            X_aligned[col] = X_test[src]

    if missing:
        log.warning(
            f'Model expected {len(missing)} missing feature columns; added as NaN. '
            f'Sample: {missing[:10]}'
        )
    return X_aligned.reindex(columns=required_cols)


def _prepare_meta(df_full: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Align metadata rows and normalize sensitive-column names expected by FairnessAuditor."""
    df_meta = df_full.iloc[:n_rows].reset_index(drop=True).copy()

    rename_map = {}
    wealth_col = _resolve_column(df_meta, ['HV270', 'hv270', 'wealth_quintile'])
    sex_col = _resolve_column(df_meta, ['B4', 'b4', 'child_sex', 'sex'])
    asp_col = _resolve_column(df_meta, ['is_aspirational', 'aspirational_flag', 'aspirational'])

    if wealth_col is not None and wealth_col != 'HV270':
        rename_map[wealth_col] = 'HV270'
    if sex_col is not None and sex_col != 'B4':
        rename_map[sex_col] = 'B4'
    if asp_col is not None and asp_col != 'is_aspirational':
        rename_map[asp_col] = 'is_aspirational'

    if rename_map:
        df_meta = df_meta.rename(columns=rename_map)

    return df_meta
 
 
def run_fairness_for_model(model, model_name, X_test, y_test, df_meta):
    """Run full fairness audit for one model. Returns equity DataFrame."""
    X_aligned = _align_features_for_model(model, X_test)
    auditor = FairnessAuditor(model, X_aligned, y_test, df_meta)
    report = auditor.build_equity_report()
    report['model'] = model_name
    return report
 
 
def main():
    print('='*65)
    print('MalnutriSense — Federated vs Centralised Fairness Comparison')
    print('='*65)
 
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    X_test = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    y_test = pd.read_csv(TRAIN_TEST_DIR / 'y_test.csv')
 
    # Load demographics for fairness audit
    scst_path = PROCESSED_DIR / 'nfhs5_with_scst.csv'
    df_full = pd.read_csv(scst_path) if scst_path.exists() else pd.read_csv(NFHS5_CLEANED_PATH)
    df_meta = _prepare_meta(df_full, len(X_test))
 
    # Load both models
    central_model = load_model(MODELS_DIR / 'mltp_xgb_v1.pkl')
    fl_model      = load_model(MODELS_DIR / 'fl_global_model.pkl')
 
    # Run fairness audit for both
    log.info('Running centralised fairness audit...')
    central_equity = run_fairness_for_model(
        central_model, 'centralised', X_test, y_test, df_meta
    )
 
    log.info('Running federated fairness audit...')
    fl_equity = run_fairness_for_model(
        fl_model, 'federated', X_test, y_test, df_meta
    )
 
    # Combine into one comparison table
    combined = pd.concat([central_equity, fl_equity], ignore_index=True)
    out_path = TABLES_DIR / 'fl_fairness_comparison.csv'
    combined.to_csv(out_path, index=False)
    log.info(f'Fairness comparison saved: {out_path}')
 
    # Create comparison plot — FNR by wealth quintile for both models
    wealth_data = combined[combined['sensitive_feature'] == 'wealth_quintile'] if not combined.empty else pd.DataFrame()
    feature_for_plot = 'wealth_quintile'
    if wealth_data.empty and not combined.empty:
        feature_for_plot = str(combined['sensitive_feature'].iloc[0])
    plot_data = combined[combined['sensitive_feature'] == feature_for_plot] if not combined.empty else pd.DataFrame()

    fig_path = FIGURES_DIR / 'fl_fnr_comparison.png'
    if not plot_data.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, label in zip(axes, TARGET_COLS):
            label_data = plot_data[plot_data['label'] == label]
            for model_name, group in label_data.groupby('model'):
                style = '-o' if model_name == 'centralised' else '--s'
                ax.plot(group['group_value'].astype(str), group['fnr'],
                        style, label=model_name, linewidth=2, markersize=6)
            ax.axhline(FNR_TOLERANCE, color='red', linestyle=':',
                       linewidth=1, label=f'FNR tolerance ({FNR_TOLERANCE:.0%})')
            ax.set_title(f'{label.capitalize()}', fontweight='bold')
            ax.set_xlabel(feature_for_plot)
            if ax == axes[0]: ax.set_ylabel('False Negative Rate')
            ax.legend(fontsize=8)
 
        plt.suptitle(f'FNR by {feature_for_plot} — Centralised vs Federated',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No fairness rows available to plot', ha='center', va='center')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
    print(f'Saved: {fig_path}')
 
    # Print summary
    print('\nFNR violation counts (FNR > 15%):')
    for model_name in ['centralised', 'federated']:
        subset = combined[(combined['model']==model_name) & (combined['fnr_exceeds_tolerance'])]
        print(f'  {model_name}: {len(subset)} subgroups exceed FNR tolerance')
 
    print(f'\nSaved: {out_path}')
 
if __name__ == '__main__':
    main()
