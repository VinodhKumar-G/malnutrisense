# MalnutriSense API Documentation
 
Base URL: `http://localhost:8000` (development) or your deployed host
Version: 1.0.0
 
## Authentication
 
All routes except `/health` require an `X-API-Key` header when
`MALNUTRISENSE_API_KEY` is set in the environment (see api/auth.py, Step 54).
 
```
X-API-Key: your-api-key-here
```
 
---
 
## GET /health
 
Health check. No authentication required. No request body.
 
**Response 200:**
```json
{
  "status": "healthy",
  "model": "mltp_xgb_v1",
  "api_version": "1.0.0"
}
```
 
**curl:**
```bash
curl http://localhost:8000/health
```
 
---
 
## POST /predict
 
Predict malnutrition risk for one child. Returns probability and binary
prediction for stunting, underweight, and wasting.
 
### Request body (ChildFeatures)
 
| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| age_months | integer | Yes | 0–59 | Child age in completed months |
| sex | string | Yes | 'male' or 'female' | Child sex |
| wealth_quintile | integer | Yes | 1–5 | 1=Poorest, 5=Richest |
| mother_education | string | Yes | no_education / primary / secondary / higher | Maternal education level |
| water_source | string | Yes | piped_on_premises / tube_well / protected_well / unprotected_well / surface_water | Primary drinking water source |
| toilet_type | string | Yes | flush_piped / pit_with_slab / pit_without_slab / other | Toilet facility type |
| diarrhoea_2weeks | integer | Yes | 0 or 1 | Diarrhoea episode in last 2 weeks |
| birth_weight_g | float | No | 500–5000 | Birth weight in grams (defaults to 3000 if omitted) |
| breastfeed_months | float | No | 0+ | Duration of breastfeeding (defaults to 6 if omitted) |
| residence | string | No | 'urban' or 'rural' | Defaults to 'rural' |
| state_code | integer | No | 1–36 | NFHS V024 state code (defaults to 9, Uttar Pradesh) |
 
### Example request
 
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{
    "age_months": 14,
    "sex": "female",
    "wealth_quintile": 1,
    "mother_education": "no_education",
    "water_source": "surface_water",
    "toilet_type": "pit_without_slab",
    "diarrhoea_2weeks": 1
  }'
```
 
### Python example
 
```python
import requests
 
resp = requests.post(
    'http://localhost:8000/predict',
    headers={'X-API-Key': 'your-key'},
    json={
        'age_months': 14, 'sex': 'female', 'wealth_quintile': 1,
        'mother_education': 'no_education', 'water_source': 'surface_water',
        'toilet_type': 'pit_without_slab', 'diarrhoea_2weeks': 1,
    },
)
print(resp.json())
```
 
### Response 200
 
```json
{
  "stunted":     {"probability": 0.78, "prediction": 1, "threshold": 0.31},
  "underweight": {"probability": 0.71, "prediction": 1, "threshold": 0.35},
  "wasted":      {"probability": 0.42, "prediction": 0, "threshold": 0.50},
  "overall_risk": "high",
  "equity_flag": true,
  "equity_reason": "Child is in the poorest two wealth quintiles — threshold correction applied to reduce missed cases."
}
```
 
### Response fields
 
| Field | Type | Description |
|---|---|---|
| stunted / underweight / wasted | object | probability (0-1), prediction (0/1), threshold applied |
| overall_risk | string | 'high', 'medium', or 'low' — derived from max probability |
| equity_flag | boolean | True if child is in a demographic group with corrected threshold |
| equity_reason | string | Explanation shown only when equity_flag is true |
 
---
 
## POST /explain
 
Same request body and response as `/predict`, with one additional field:
`top_shap_features` — the top 3 features driving the highest-risk phenotype's
prediction for this specific child.
 
### Additional response field
 
```json
{
  ...(all /predict fields)...
  "top_shap_features": [
    {"feature": "HV270", "shap_value": 0.42,  "value": "1"},
    {"feature": "V106",  "shap_value": 0.31,  "value": "no_education"},
    {"feature": "HV201", "shap_value": 0.18,  "value": "surface_water"}
  ]
}
```
 
`shap_value` is positive when the feature pushes the prediction toward
malnutrition risk, and negative when it protects against it.
 
---
 
## Error Codes
 
| Code | Meaning | Cause |
|---|---|---|
| 401 | Unauthorized | Missing X-API-Key header (when auth is enabled) |
| 403 | Forbidden | X-API-Key header present but incorrect |
| 422 | Unprocessable Entity | Request body fails Pydantic validation (e.g. age_months > 59) |
| 500 | Internal Server Error | Unexpected prediction failure — check server logs |
| 503 | Service Unavailable | Model not yet loaded (API still starting up) |
 
### Example 422 response
 
```json
{
  "detail": [
    {
      "loc": ["body", "age_months"],
      "msg": "ensure this value is less than or equal to 59",
      "type": "value_error.number.not_le"
    }
  ]
}
```
