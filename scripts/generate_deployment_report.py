"""
scripts/generate_deployment_report.py — Phase 6 deployment completion report.
 
Sections:
  1. API Architecture — routes, authentication, model version
  2. Dashboard Features — pages, capabilities
  3. Docker Deployment — containerisation status
  4. Test Results — API tests, recall verification, browser checklist
 
Writes: reports/deployment_report.txt
 
Usage: python3 scripts/generate_deployment_report.py
"""
 
import sys, subprocess
from datetime import datetime, timezone
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
REPORT_PATH = Path('reports/deployment_report.txt')
 
 
def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
 
def _div(path, char='─', w=70):
    _w(path, char * w + '\n')
 
 
def main():
    REPORT_PATH.unlink(missing_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
 
    _w(REPORT_PATH, '\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, 'MALNUTRISENSE — PHASE 6 DEPLOYMENT REPORT\n')
    _w(REPORT_PATH, f'Generated: {ts}\n')
    _div(REPORT_PATH, '=')
 
    # Section 1: API Architecture
    _w(REPORT_PATH, '\nSECTION 1: API ARCHITECTURE\n')
    _div(REPORT_PATH)
    _w(REPORT_PATH, '  Framework:       FastAPI\n')
    _w(REPORT_PATH, '  Routes:          GET /health, POST /predict, POST /explain\n')
    _w(REPORT_PATH, '  Authentication:  X-API-Key header (api/auth.py)\n')
    _w(REPORT_PATH, '  Model served:    mltp_xgb_v1.pkl\n')
    _w(REPORT_PATH, '  Threshold logic: Per-subgroup corrected thresholds '
       '(reports/tables/corrected_thresholds.json)\n')
 
    api_exists = Path('api/main.py').exists()
    _w(REPORT_PATH, f'  Status: {"DEPLOYED" if api_exists else "NOT FOUND"}\n')
 
    # Section 2: Dashboard Features
    _w(REPORT_PATH, '\nSECTION 2: DASHBOARD FEATURES\n')
    _div(REPORT_PATH)
    pages = {
        'dashboard/app.py':                  'Main prediction interface',
        'dashboard/pages/batch_predict.py':  'CSV batch upload and prediction',
        'dashboard/pages/about.py':          'SHAP and equity explanation guide',
    }
    for path, desc in pages.items():
        status = 'OK' if Path(path).exists() else 'MISSING'
        _w(REPORT_PATH, f'  [{status}] {path:<40} {desc}\n')
 
    # Section 3: Docker Deployment
    _w(REPORT_PATH, '\nSECTION 3: DOCKER DEPLOYMENT\n')
    _div(REPORT_PATH)
    docker_files = ['Dockerfile', 'docker-compose.yml']
    for f in docker_files:
        status = 'OK' if Path(f).exists() else 'MISSING'
        _w(REPORT_PATH, f'  [{status}] {f}\n')
 
    # Section 4: Test Results
    _w(REPORT_PATH, '\nSECTION 4: TEST RESULTS\n')
    _div(REPORT_PATH)
 
    # API unit/integration tests
    try:
        result = subprocess.run(
            ['pytest', 'tests/test_api.py', '-q', '--tb=no'],
            capture_output=True, text=True, timeout=60
        )
        _w(REPORT_PATH, f'  API integration tests: {result.stdout.strip().splitlines()[-1] if result.stdout else "N/A"}\n')
    except Exception as e:
        _w(REPORT_PATH, f'  API integration tests: could not run automatically ({e})\n')
        _w(REPORT_PATH, '    Run manually: pytest tests/test_api.py -v\n')
 
    # Recall verification
    recall_note = (
        '  Recall verification: run scripts/verify_api_recall.py with API live\n'
        '    (requires uvicorn running — not run automatically by this report)\n'
    )
    _w(REPORT_PATH, recall_note)
 
    # Browser checklist
    checklist_path = Path('reports/testing_checklist.md')
    if checklist_path.exists():
        content = checklist_path.read_text()
        pass_count = content.count('[x] PASS') + content.count('[X] PASS')
        fail_count = content.count('[x] FAIL') + content.count('[X] FAIL')
        _w(REPORT_PATH, f'  Browser checklist: {pass_count} passed, {fail_count} failed '
           f'(see {checklist_path})\n')
    else:
        _w(REPORT_PATH, '  Browser checklist: not yet generated (Step 60)\n')
 
    _w(REPORT_PATH, '\n')
    _div(REPORT_PATH, '=')
    _w(REPORT_PATH, 'END OF DEPLOYMENT REPORT\n')
    _div(REPORT_PATH, '=')
 
    print(f'Deployment report saved: {REPORT_PATH}')
    print(REPORT_PATH.read_text())
 
 
if __name__ == '__main__':
    main()
