

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

        self.model_path = model_path
        self.model = load_model(model_path)
        log.info('MLTP model loaded')

        thresh_path = TABLES_DIR / 'corrected_thresholds.json'
        self.thresholds = json.loads(thresh_path.read_text()) if thresh_path.exists() else {}
        log.info(f'Thresholds loaded: {len(self.thresholds)} labels')

        # Load class weights for equity flag
        self.class_weights = json.loads((TABLES_DIR/'class_weights.json').read_text())

    def _extract_required_feature_columns(self) -> list[str]:
        """Return feature columns expected by model preprocessor."""
        pre = getattr(self.model, 'named_steps', {}).get('pre') if hasattr(self.model, 'named_steps') else None
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
 
    def features_to_df(self, features) -> pd.DataFrame:
        """Convert ChildFeatures schema to DataFrame row."""
        sex_map = {'male': 1, 'female': 2}
        residence_map = {'urban': 1, 'rural': 2}
        education_map = {
            'no_education': 0,
            'primary': 1,
            'secondary': 2,
            'higher': 3,
        }

        sex_val = sex_map.get(str(features.sex).strip().lower(), np.nan)
        residence_val = residence_map.get(str(features.residence).strip().lower(), np.nan)
        edu_val = education_map.get(str(features.mother_education).strip().lower(), np.nan)

        weight_kg = getattr(features, 'weight_kg', None)
        height_cm = getattr(features, 'height_cm', None)

        # Use lowercase DHS-style names to match model training schema.
        row = {
            'v024': features.state_code or 9,
            'v025': residence_val,
            'v106': edu_val,
            'v130': np.nan,  # not provided by request schema
            'b4': sex_val,
            'm4': features.breastfeed_months if features.breastfeed_months is not None else 6,
            'm19': features.birth_weight_g if features.birth_weight_g is not None else 3000,
            'h11': features.diarrhoea_2weeks,
            'hw1': features.age_months,
            'hw2': (float(weight_kg) * 10) if weight_kg is not None else np.nan,
            'hw3': (float(height_cm) * 10) if height_cm is not None else np.nan,
        }

        X = pd.DataFrame([row])
        required_cols = self._extract_required_feature_columns()
        if required_cols:
            for col in required_cols:
                if col not in X.columns:
                    X[col] = np.nan
            X = X.reindex(columns=required_cols)
        return X

    def debug_prediction_case(self, features, reference_profile=None) -> dict:
        """Return a structured investigation for unexpected high-risk predictions."""
        X = self.features_to_df(features)
        proba_list = self.model.predict_proba(X)
        probabilities = {
            label: float(proba_list[idx][0, 1])
            for idx, label in enumerate(TARGET_COLS)
        }

        first_result = self.predict(features)
        second_result = self.predict(features)
        reproducible = first_result == second_result

        feature_names = list(X.columns)
        pre = getattr(self.model, 'named_steps', {}).get('pre', None)
        if pre is not None and hasattr(pre, 'get_feature_names_out'):
            feature_names = list(pre.get_feature_names_out())

        investigation = {
            'reproducible': reproducible,
            'probabilities': probabilities,
            'feature_values': X.iloc[0].to_dict(),
            'feature_order': list(X.columns),
            'feature_names': feature_names,
            'missing_features': {k: str(v) for k, v in X.iloc[0].items() if pd.isna(v)},
            'model_artifact': str(self.model_path),
            'prediction_1': first_result,
            'prediction_2': second_result,
        }

        if reference_profile is not None:
            reference_result = self.predict(reference_profile)
            if isinstance(reference_result, dict):
                reference_max = max(reference_result[label]['probability'] for label in TARGET_COLS)
                current_max = max(first_result[label]['probability'] for label in TARGET_COLS)
                investigation['reference_comparison'] = {
                    'reference_profile': reference_profile,
                    'reference_prediction': reference_result,
                    'current_max_probability': current_max,
                    'reference_max_probability': reference_max,
                    'current_not_lower_than_reference': current_max >= reference_max,
                }

        return investigation
 
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
 
