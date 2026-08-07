"""
api/auth.py — Simple API key authentication middleware.
 
Reads API key from environment variable MALNUTRISENSE_API_KEY.
Validates X-API-Key header on /predict and /explain endpoints.
The /health endpoint is public (no auth required).
"""
 
import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
 
API_KEY_NAME = 'X-API-Key'
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
 
# Default key for development — override with env var in production
API_KEY = os.getenv('MALNUTRISENSE_API_KEY', 'dev-key-change-in-production')
 
 
async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate the X-API-Key header."""
    if api_key is None:
        raise HTTPException(status_code=401, detail='Missing X-API-Key header')
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail='Invalid API key')
    return api_key
 
 
# Usage in api/main.py:
# from api.auth import verify_api_key
# @app.post('/predict', dependencies=[Depends(verify_api_key)])
# async def predict(features: ChildFeatures):
#     ...
 
