"""
scripts/check_environment.py — Verify all required packages are installed.
 
Run after a fresh environment setup to confirm nothing is missing
before attempting to run Steps 1-63.
 
Usage: python3 scripts/check_environment.py
"""
 
import importlib
import sys
 
REQUIRED_PACKAGES = {
    'pandas': 'Data pipeline (Steps 1-18)',
    'numpy': 'Data pipeline (Steps 1-18)',
    'pyreadstat': 'DHS .DTA file loading (Step 8)',
    'sklearn': 'Model pipeline (Steps 19-24)',
    'xgboost': 'MLTP primary model (Step 19)',
    'lightgbm': 'MLTP comparison model (Step 19)',
    'shap': 'Explainability (Step 25)',
    'fairlearn': 'Fairness audit (Step 26)',
    'geopandas': 'Choropleth map (Step 30)',
    'ee': 'Google Earth Engine (Steps 31-32)',
    'flwr': 'Federated learning (Steps 38-45)',
    'fastapi': 'API (Steps 46-48)',
    'uvicorn': 'API server (Step 48)',
    'streamlit': 'Dashboard (Step 49)',
    'pytest': 'All test suites',
}
 
def main():
    print('Checking MalnutriSense environment...')
    print('='*60)
    missing = []
    for package, purpose in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f'  [OK] {package:<15} v{version:<12} — {purpose}')
        except ImportError:
            print(f'  [MISSING] {package:<15} — needed for: {purpose}')
            missing.append(package)
 
    print('='*60)
    if missing:
        print(f'{len(missing)} package(s) missing. Install with:')
        print(f'  pip install {" ".join(missing)}')
        sys.exit(1)
    else:
        print('All required packages installed. Environment ready.')
 
if __name__ == '__main__':
    main()
