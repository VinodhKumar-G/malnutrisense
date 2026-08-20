"""
scripts/generate_testing_checklist.py — Scaffold the cross-browser testing checklist.
 
Run once to create reports/testing_checklist.md with a fixed set of test cases.
Testers then manually open the dashboard in each browser and fill in PASS/FAIL.
 
Usage: python3 scripts/generate_testing_checklist.py
"""
 
from pathlib import Path
from datetime import datetime, timezone
 
REPORT_PATH = Path('reports/testing_checklist.md')
 
TEST_CASES = [
    'Dashboard loads without console errors',
    'All form inputs (sliders, dropdowns, number input) render correctly',
    'Predict button triggers a request and shows a loading spinner',
    'Risk score cards (3 metrics) display after prediction',
    'Overall risk banner shows correct colour (red/yellow/green)',
    'SHAP bar chart renders with correct red/green bar colours',
    'Equity flag message appears for wealth_quintile=1 inputs',
    'Equity flag message does NOT appear for wealth_quintile=5 inputs',
    'Batch prediction page: CSV upload widget accepts a .csv file',
    'Batch prediction page: results table renders after processing',
    'Batch prediction page: download button produces a valid CSV',
    'About page renders all markdown formatting correctly',
    'Page is usable at mobile width (375px) without horizontal scroll',
    'Page is usable at tablet width (768px)',
]
 
BROWSERS = ['Chrome (desktop)', 'Firefox (desktop)', 'Safari (desktop, if available)',
            'Chrome (mobile emulation, 375px)', 'Chrome (tablet emulation, 768px)']
 
def main():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [
        '# MalnutriSense Dashboard — Cross-Browser Testing Checklist',
        f'',
        f'Generated: {ts}',
        f'Dashboard URL under test: http://localhost:8501',
        f'API URL required: http://localhost:8000 (start with `bash scripts/start_services.sh`',
        f'or `uvicorn api.main:app --host 0.0.0.0 --port 8000` before opening the dashboard —',
        f'Predict/Batch Predict rows will fail if the API is not running)',
        f'',
        f'Instructions: Open the dashboard in each browser below. For each test',
        f'case, mark PASS or FAIL. Use Chrome DevTools device toolbar (Ctrl+Shift+M)',
        f'for the mobile and tablet rows.',
        f'',
    ]
    for browser in BROWSERS:
        lines.append(f'## {browser}')
        lines.append('')
        lines.append('| # | Test Case | Result | Notes |')
        lines.append('|---|---|---|---|')
        for i, case in enumerate(TEST_CASES, 1):
            lines.append(f'| {i} | {case} | [ ] PASS / [ ] FAIL | |')
        lines.append('')
 
    lines.append('## Summary')
    lines.append('')
    lines.append('Total test cases: ' + str(len(TEST_CASES) * len(BROWSERS)))
    lines.append('Passed: ___')
    lines.append('Failed: ___')
    lines.append('')
    lines.append('## Issues found (if any)')
    lines.append('')
    lines.append('- ')
 
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Checklist scaffolded: {REPORT_PATH}')
    print(f'{len(TEST_CASES)} test cases x {len(BROWSERS)} browsers = '
          f'{len(TEST_CASES)*len(BROWSERS)} total checks')
    print('IMPORTANT: start the API first — bash scripts/start_services.sh')
    print('(or `uvicorn api.main:app --host 0.0.0.0 --port 8000`), then the dashboard.')
    print('Open the dashboard in each browser and fill in PASS/FAIL manually.')
 
if __name__ == '__main__':
    main()
