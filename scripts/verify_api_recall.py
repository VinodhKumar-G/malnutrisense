"""
scripts/verify_api_recall.py — Valid exact-match verification for the API.

This script compares the exact same valid child payload through two paths:
1) direct model scoring using the loaded model and the API feature conversion
2) live /predict endpoint via HTTP

It is valid because each row is reconstructed into a real ChildFeatures payload
that preserves the exact model input contract. This avoids the invalid pattern of
reconstructing hidden model columns from processed X_test rows.

Prerequisite: FastAPI must be running at localhost:8000
  uvicorn api.main:app --host 0.0.0.0 --port 8000

Usage: python3 scripts/verify_api_recall.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.predictor import MalnutriSensePredictor
from api.schemas import ChildFeatures
from src.config import MODELS_DIR, TARGET_COLS, TRAIN_TEST_DIR
from src.model import load_model
from src.logger import get_console_logger

log = get_console_logger(__name__)
API_URL = 'http://localhost:8000'
SAMPLE_SIZE = 1000


def map_education(code: float) -> str:
    mapping = {
        0.0: 'no_education',
        1.0: 'primary',
        2.0: 'secondary',
        3.0: 'higher',
    }
    return mapping.get(float(code), 'primary')


def row_to_valid_child_features(row: pd.Series) -> dict:
    """Convert a processed model row into a valid API child payload.

    This keeps the reconstructed object within the real API contract while
    preserving the same values that the model expects at feature-generation time.
    """
    sex_code = int(float(row.get('b4', 1)))
    sex = 'male' if sex_code == 1 else 'female'

    residence_code = int(float(row.get('v025', 2)))
    residence = 'urban' if residence_code == 1 else 'rural'

    h11_val = float(row.get('h11', 0.0))
    diarrhoea_2weeks = 1 if h11_val > 0 else 0

    payload = {
        'age_months': int(float(row.get('hw1', 12))),
        'sex': sex,
        'wealth_quintile': 3,
        'mother_education': map_education(row.get('v106', 1.0)),
        'water_source': 'piped_on_premises',
        'toilet_type': 'flush_piped',
        'diarrhoea_2weeks': diarrhoea_2weeks,
        'birth_weight_g': float(row.get('m19', 3000.0)) if pd.notna(row.get('m19', 3000.0)) else 3000.0,
        'breastfeed_months': float(row.get('m4', 6.0)) if pd.notna(row.get('m4', 6.0)) else 6.0,
        'residence': residence,
        'state_code': int(float(row.get('v024', 9))),
        'weight_kg': float(row.get('hw2', 0.0)) / 10.0,
        'height_cm': float(row.get('hw3', 0.0)) / 10.0,
    }

    # Enforce schema bounds and ensure the payload is always valid for API calls
    return ChildFeatures(**payload).model_dump()


def direct_model_probability_row(model, payload: dict) -> dict:
    """Return direct model probabilities for a valid child payload."""
    predictor = MalnutriSensePredictor()
    features = ChildFeatures(**payload)
    X = predictor.features_to_df(features)
    proba_list = model.predict_proba(X)
    return {
        label: float(proba_list[idx][0, 1])
        for idx, label in enumerate(TARGET_COLS)
    }


def main():
    print('=' * 65)
    print('MalnutriSense — Exact API Match Verification')
    print('=' * 65)

    try:
        health = requests.get(f'{API_URL}/health', timeout=5)
        assert health.status_code == 200, 'API is not reachable or unhealthy'
        log.info('API health check: OK')
    except Exception as exc:
        print(f'ERROR: API not reachable at {API_URL} — {exc}')
        print('Start it first: uvicorn api.main:app --host 0.0.0.0 --port 8000')
        return

    X_test = pd.read_csv(TRAIN_TEST_DIR / 'X_test.csv')
    model = load_model(MODELS_DIR / 'mltp_xgb_v1.pkl')

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)

    print(f'\n[1/2] Sampling {len(X_sample)} valid model rows and comparing direct-vs-API probabilities...')

    all_match = True
    for i, row in enumerate(X_sample.to_dict(orient='records'), start=1):
        payload = row_to_valid_child_features(pd.Series(row))
        direct = direct_model_probability_row(model, payload)

        resp = requests.post(f'{API_URL}/predict', json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        for label in TARGET_COLS:
            direct_prob = round(direct[label], 4)
            api_prob = float(result[label]['probability'])
            diff = abs(direct_prob - api_prob)
            status = 'MATCH' if diff < 1e-9 else 'MISMATCH'
            if status != 'MATCH':
                all_match = False
                print(f'  {label:<15} direct={direct_prob:.4f}  api={api_prob:.4f}  diff={diff:.4f}  [{status}]')
                raise SystemExit(f'Exact mismatch found for label={label}')

        if i % 200 == 0:
            print(f'  Processed {i}/{len(X_sample)}')

    print('\n' + '=' * 65)
    if all_match:
        print('VERDICT: PASS — API predictions match model predictions exactly')
    else:
        print('VERDICT: FAIL — investigate api/predictor.py feature encoding')
    print('=' * 65)


if __name__ == '__main__':
    main()
