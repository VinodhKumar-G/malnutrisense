"""
api/main.py — FastAPI application for MalnutriSense.
 
Routes:
  GET  /health   — health check
  POST /predict  — predict malnutrition risk for one child
  POST /explain  — predict + return top-3 SHAP features
 
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
 
from api.schemas import ChildFeatures, PredictionResponse
from api.predictor import MalnutriSensePredictor
from src.logger import get_console_logger
 
log = get_console_logger('api.main')
 
app = FastAPI(
    title='MalnutriSense API',
    description='Multi-label child malnutrition risk prediction with SHAP explanations',
    version='1.0.0',
)
 
# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
 
# Load model once at startup
predictor = None
 
@app.on_event('startup')
async def startup_event():
    global predictor
    predictor = MalnutriSensePredictor()
    log.info('MalnutriSense API started')
 
 
@app.get('/health')
def health_check():
    """Returns API status and model version."""
    return {
        'status': 'healthy',
        'model':  'mltp_xgb_v1',
        'api_version': '1.0.0',
    }


@app.get('/')
def root():
    """Root route for browser/open-link checks."""
    return health_check()
 
 
@app.post('/predict', response_model=dict)
async def predict(features: ChildFeatures):
    """
    Predict malnutrition risk for a single child.
    Returns probability and prediction (0/1) for stunting, underweight, wasting.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        result = predictor.predict(features)
        return result
    except Exception as e:
        log.error(f'/predict error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/predict')
def predict_get_help():
    """Browser-friendly hint for POST-only predict route."""
    return {
        'message': 'Use POST /predict with JSON body.',
        'docs': '/docs',
        'example_curl': "curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"age_months\":12,\"sex\":\"female\",\"wealth_quintile\":1,\"mother_education\":\"no_education\",\"water_source\":\"surface_water\",\"toilet_type\":\"pit_without_slab\",\"diarrhoea_2weeks\":1,\"residence\":\"rural\"}'",
    }
 
 
@app.post('/explain', response_model=dict)
async def explain(features: ChildFeatures):
    """
    Predict + return top-3 SHAP features for the prediction.
    Used by the Streamlit dashboard to show actionable risk factors.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        prediction = predictor.predict(features)
        explanation = predictor.explain(features)
        return {**prediction, 'top_shap_features': explanation}
    except Exception as e:
        log.error(f'/explain error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/explain')
def explain_get_help():
    """Browser-friendly hint for POST-only explain route."""
    return {
        'message': 'Use POST /explain with JSON body.',
        'docs': '/docs',
    }
 
 
if __name__ == '__main__':
    uvicorn.run('api.main:app', host='0.0.0.0', port=8000, reload=True)
 
