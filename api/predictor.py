

"""
api/predictor.py — Core prediction logic used by the FastAPI endpoints.
 
Loads the MLTP model and SHAP explainer at startup (not per-request).
predict() returns risk scores per phenotype.
explain() returns top-3 SHAP features for the prediction.
"""
 
import json
import numpy as np
import pandas as pd
from pathlib import Path
 
from src.model import load_model
from src.config import MODELS_DIR, TABLES_DIR, TARGET_COLS
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
 
class MalnutriSensePredictor:
    """Singleton predictor — loaded once at API startup."""
 
    def __init__(self):
        model_path = MODELS_DIR / 'mltp_xgb_v1.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path}')
 
        self.model = load_model(model_path)
        log.info('MLTP model loaded')
 
        # Load corrected thresholds for equity-aware prediction
        thresh_path = TABLES_DIR / 'corrected_thresholds.json'
        self.thresholds = json.loads(thresh_path.read_text()) if thresh_path.exists() else {}
        log.info(f'Thresholds loaded: {len(self.thresholds)} labels')
 
        # Load class weights for equity flag
        self.class_weights = json.loads((TABLES_DIR/'class_weights.json').read_text())
 
    def features_to_df(self, features) -> pd.DataFrame:
        """Convert ChildFeatures schema to DataFrame row."""
        return pd.DataFrame([{
            'HW1':   features.age_months,
            'B4':    features.sex,
            'HV270': features.wealth_quintile,
            'V106':  features.mother_education,
            'HV201': features.water_source,
            'HV205': features.toilet_type,
            'H11':   features.diarrhoea_2weeks,
            'M19':   features.birth_weight_g or 3000,
            'M4':    features.breastfeed_months or 6,
            'V025':  features.residence,
            'V024':  features.state_code or 9,
            'sdi':   (5 - features.wealth_quintile) / 4.0,  # simplified SDI
        }])
 
    def predict(self, features) -> dict:
        """Return per-phenotype probability and prediction."""
        X = self.features_to_df(features)
        proba_list = self.model.predict_proba(X)
 
        results = {}
        for i, label in enumerate(TARGET_COLS):
            prob = float(proba_list[i][0, 1])
 
            # Use corrected threshold if available for this wealth quintile
            default_thresh = 0.5
            wealth_key = str(features.wealth_quintile)
            thresh = self.thresholds.get(label, {}).get(wealth_key, default_thresh)
 
            results[label] = {
                'probability': round(prob, 4),
                'prediction':  int(prob >= thresh),
                'threshold':   thresh,
            }
 
        # Overall risk level
        max_prob = max(r['probability'] for r in results.values())
        if max_prob >= 0.65:   overall_risk = 'high'
        elif max_prob >= 0.40: overall_risk = 'medium'
        else:                   overall_risk = 'low'
        results['overall_risk'] = overall_risk
 
        # Equity flag — wealth quintile 1 or 2
        results['equity_flag'] = features.wealth_quintile <= 2
        results['equity_reason'] = (
            'Child is in the poorest two wealth quintiles — '
            'threshold correction applied to reduce missed cases.'
            if features.wealth_quintile <= 2 else ''
        )
        return results
 
    def explain(self, features) -> list:
        """Return top-3 SHAP features for the prediction."""
        try:
            import shap
            from src.explainability import SHAPExplainer
            X = self.features_to_df(features)
 
            # Use a small background for fast API-time SHAP
            explainer = SHAPExplainer(self.model, X)  # X as its own background
            shap_vals = explainer.compute_shap_values(X)
 
            # Get top 3 features by absolute SHAP for the highest-risk label
            max_label_idx = 0  # stunted by default
            sv = shap_vals[max_label_idx][0]
            feature_names = explainer.feature_names
 
            top3_idx = np.argsort(np.abs(sv))[::-1][:3]
            return [
                {'feature': feature_names[i],
                 'shap_value': round(float(sv[i]), 4),
                 'value': str(X.iloc[0, i] if i < X.shape[1] else '')}
                for i in top3_idx
            ]
        except Exception as e:
            log.warning(f'SHAP explain failed: {e}')
            return []
 
