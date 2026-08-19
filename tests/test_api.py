"""
tests/test_api.py — Integration tests for the FastAPI prediction API.
 
Tests: /health, /predict with valid data, /predict with invalid data,
       /explain, response schema validation, equity flag logic.
 
Run: pytest tests/test_api.py -v
Note: Requires uvicorn NOT running — tests use TestClient which runs in-process.
"""
 
import pytest
from fastapi.testclient import TestClient
 
 
@pytest.fixture(scope='module')
def client():
    from api.main import app
    return TestClient(app)
 
 
VALID_CHILD = {
    'age_months': 12, 'sex': 'female', 'wealth_quintile': 1,
    'mother_education': 'no_education', 'water_source': 'surface_water',
    'toilet_type': 'pit_without_slab', 'diarrhoea_2weeks': 1,
    'birth_weight_g': 2200, 'breastfeed_months': 3, 'residence': 'rural',
}

LOW_RISK_CHILD = {
    'age_months': 36,
    'sex': 'male',
    'wealth_quintile': 5,
    'mother_education': 'higher',
    'water_source': 'piped_on_premises',
    'toilet_type': 'flush_piped',
    'diarrhoea_2weeks': 0,
    'birth_weight_g': 3500,
    'breastfeed_months': 12,
    'residence': 'urban',
    'weight_kg': 14.5,
    'height_cm': 95.0,
}

HIGH_RISK_CHILD = {
    'age_months': 36,
    'sex': 'male',
    'wealth_quintile': 1,
    'mother_education': 'none',
    'water_source': 'surface',
    'toilet_type': 'no_facility',
    'diarrhoea_2weeks': 1,
    'birth_weight_g': 2200,
    'breastfeed_months': 0,
    'residence': 'rural',
    'weight_kg': 8.5,
    'height_cm': 72.0,
}
 
HEALTHY_CHILD = {
    'age_months': 36, 'sex': 'male', 'wealth_quintile': 5,
    'mother_education': 'higher', 'water_source': 'piped_on_premises',
    'toilet_type': 'flush_piped', 'diarrhoea_2weeks': 0,
    'birth_weight_g': 3500, 'breastfeed_months': 12, 'residence': 'urban',
}
 
 
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
 
    def test_health_contains_status(self, client):
        data = client.get('/health').json()
        assert data['status'] == 'healthy'
        assert 'model' in data
 
 
class TestPredictEndpoint:
    def test_low_risk_child_validation_workflow(self, client):
        from api.predictor import MalnutriSensePredictor
        from api.schemas import ChildFeatures

        predictor = MalnutriSensePredictor()
        resp = client.post('/predict', json=LOW_RISK_CHILD)
        assert resp.status_code == 200
        result = resp.json()

        assert result['overall_risk'] in ['low', 'medium']
        if result['overall_risk'] == 'high':
            debug = predictor.debug_prediction_case(ChildFeatures(**LOW_RISK_CHILD), reference_profile=ChildFeatures(**HIGH_RISK_CHILD))
            assert debug['reproducible'] is True
            assert set(debug['probabilities']) == {'stunted', 'underweight', 'wasted'}
            assert set(debug['feature_order']) == {'v024', 'v025', 'v106', 'v130', 'b4', 'm4', 'm19', 'h11', 'hw1', 'hw2', 'hw3'}
            assert 'reference_comparison' in debug
            assert debug['reference_comparison']['current_not_lower_than_reference'] is True

    def test_high_risk_child_is_not_lower_than_low_risk(self, client):
        low_resp = client.post('/predict', json=LOW_RISK_CHILD)
        high_resp = client.post('/predict', json=HIGH_RISK_CHILD)
        low_result = low_resp.json()
        high_result = high_resp.json()

        low_max = max(low_result[label]['probability'] for label in ['stunted', 'underweight', 'wasted'])
        high_max = max(high_result[label]['probability'] for label in ['stunted', 'underweight', 'wasted'])
        assert high_max >= low_max

    def test_predict_returns_200(self, client):
        resp = client.post('/predict', json=VALID_CHILD)
        assert resp.status_code == 200
 
    def test_predict_has_three_phenotypes(self, client):
        data = client.post('/predict', json=VALID_CHILD).json()
        assert 'stunted' in data
        assert 'underweight' in data
        assert 'wasted' in data
 
    def test_predict_probabilities_in_range(self, client):
        data = client.post('/predict', json=VALID_CHILD).json()
        for label in ['stunted', 'underweight', 'wasted']:
            assert 0.0 <= data[label]['probability'] <= 1.0
            assert data[label]['prediction'] in [0, 1]
            assert 0.0 <= data[label]['threshold'] <= 1.0

    def test_predict_overall_risk_level(self, client):
        data = client.post('/predict', json=VALID_CHILD).json()
        assert data['overall_risk'] in ['high', 'medium', 'low']

    def test_predict_contract_has_required_fields(self, client):
        data = client.post('/predict', json=VALID_CHILD).json()
        assert 'stunted' in data and 'underweight' in data and 'wasted' in data
        assert data['overall_risk'] in {'low', 'medium', 'high'}
        assert isinstance(data['equity_flag'], bool)
        assert isinstance(data['equity_reason'], str)
    def test_no_equity_flag_for_richest(self, client):
        data = client.post('/predict', json=HEALTHY_CHILD).json()
        assert data['equity_flag'] is False  # wealth_quintile=5 → no flag
 
    def test_invalid_age_returns_422(self, client):
        bad = {**VALID_CHILD, 'age_months': 100}  # max is 59
        resp = client.post('/predict', json=bad)
        assert resp.status_code == 422
 
    def test_missing_required_field_returns_422(self, client):
        incomplete = {'age_months': 12, 'sex': 'male'}  # missing many fields
        resp = client.post('/predict', json=incomplete)
        assert resp.status_code == 422
 
 
class TestExplainEndpoint:
    def test_explain_returns_200(self, client):
        resp = client.post('/explain', json=VALID_CHILD)
        assert resp.status_code == 200
 
    def test_explain_has_shap_features(self, client):
        data = client.post('/explain', json=VALID_CHILD).json()
        assert 'top_shap_features' in data
        # Should have up to 3 features
        assert len(data['top_shap_features']) <= 3
 
    def test_explain_shap_feature_has_required_keys(self, client):
        data = client.post('/explain', json=VALID_CHILD).json()
        if data['top_shap_features']:
            feat = data['top_shap_features'][0]
            assert 'feature' in feat
            assert 'shap_value' in feat
 
