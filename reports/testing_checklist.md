# MalnutriSense Dashboard — Cross-Browser Testing Checklist

Generated: 2026-08-20 15:16:48 UTC
Dashboard URL under test: http://localhost:8501
API URL required: http://localhost:8000 (start with `bash scripts/start_services.sh`
or `uvicorn api.main:app --host 0.0.0.0 --port 8000` before opening the dashboard —
Predict/Batch Predict rows will fail if the API is not running)

Instructions: Open the dashboard in each browser below. For each test
case, mark PASS or FAIL. Use Chrome DevTools device toolbar (Ctrl+Shift+M)
for the mobile and tablet rows.

## Chrome (desktop)

| # | Test Case | Result | Notes |
|---|---|---|---|
| 1 | Dashboard loads without console errors | [ ] PASS / [ ] FAIL | |
| 2 | All form inputs (sliders, dropdowns, number input) render correctly | [ ] PASS / [ ] FAIL | |
| 3 | Predict button triggers a request and shows a loading spinner | [ ] PASS / [ ] FAIL | |
| 4 | Risk score cards (3 metrics) display after prediction | [ ] PASS / [ ] FAIL | |
| 5 | Overall risk banner shows correct colour (red/yellow/green) | [ ] PASS / [ ] FAIL | |
| 6 | SHAP bar chart renders with correct red/green bar colours | [ ] PASS / [ ] FAIL | |
| 7 | Equity flag message appears for wealth_quintile=1 inputs | [ ] PASS / [ ] FAIL | |
| 8 | Equity flag message does NOT appear for wealth_quintile=5 inputs | [ ] PASS / [ ] FAIL | |
| 9 | Batch prediction page: CSV upload widget accepts a .csv file | [ ] PASS / [ ] FAIL | |
| 10 | Batch prediction page: results table renders after processing | [ ] PASS / [ ] FAIL | |
| 11 | Batch prediction page: download button produces a valid CSV | [ ] PASS / [ ] FAIL | |
| 12 | About page renders all markdown formatting correctly | [ ] PASS / [ ] FAIL | |
| 13 | Page is usable at mobile width (375px) without horizontal scroll | [ ] PASS / [ ] FAIL | |
| 14 | Page is usable at tablet width (768px) | [ ] PASS / [ ] FAIL | |

## Firefox (desktop)

| # | Test Case | Result | Notes |
|---|---|---|---|
| 1 | Dashboard loads without console errors | [ ] PASS / [ ] FAIL | |
| 2 | All form inputs (sliders, dropdowns, number input) render correctly | [ ] PASS / [ ] FAIL | |
| 3 | Predict button triggers a request and shows a loading spinner | [ ] PASS / [ ] FAIL | |
| 4 | Risk score cards (3 metrics) display after prediction | [ ] PASS / [ ] FAIL | |
| 5 | Overall risk banner shows correct colour (red/yellow/green) | [ ] PASS / [ ] FAIL | |
| 6 | SHAP bar chart renders with correct red/green bar colours | [ ] PASS / [ ] FAIL | |
| 7 | Equity flag message appears for wealth_quintile=1 inputs | [ ] PASS / [ ] FAIL | |
| 8 | Equity flag message does NOT appear for wealth_quintile=5 inputs | [ ] PASS / [ ] FAIL | |
| 9 | Batch prediction page: CSV upload widget accepts a .csv file | [ ] PASS / [ ] FAIL | |
| 10 | Batch prediction page: results table renders after processing | [ ] PASS / [ ] FAIL | |
| 11 | Batch prediction page: download button produces a valid CSV | [ ] PASS / [ ] FAIL | |
| 12 | About page renders all markdown formatting correctly | [ ] PASS / [ ] FAIL | |
| 13 | Page is usable at mobile width (375px) without horizontal scroll | [ ] PASS / [ ] FAIL | |
| 14 | Page is usable at tablet width (768px) | [ ] PASS / [ ] FAIL | |

## Safari (desktop, if available)

| # | Test Case | Result | Notes |
|---|---|---|---|
| 1 | Dashboard loads without console errors | [ ] PASS / [ ] FAIL | |
| 2 | All form inputs (sliders, dropdowns, number input) render correctly | [ ] PASS / [ ] FAIL | |
| 3 | Predict button triggers a request and shows a loading spinner | [ ] PASS / [ ] FAIL | |
| 4 | Risk score cards (3 metrics) display after prediction | [ ] PASS / [ ] FAIL | |
| 5 | Overall risk banner shows correct colour (red/yellow/green) | [ ] PASS / [ ] FAIL | |
| 6 | SHAP bar chart renders with correct red/green bar colours | [ ] PASS / [ ] FAIL | |
| 7 | Equity flag message appears for wealth_quintile=1 inputs | [ ] PASS / [ ] FAIL | |
| 8 | Equity flag message does NOT appear for wealth_quintile=5 inputs | [ ] PASS / [ ] FAIL | |
| 9 | Batch prediction page: CSV upload widget accepts a .csv file | [ ] PASS / [ ] FAIL | |
| 10 | Batch prediction page: results table renders after processing | [ ] PASS / [ ] FAIL | |
| 11 | Batch prediction page: download button produces a valid CSV | [ ] PASS / [ ] FAIL | |
| 12 | About page renders all markdown formatting correctly | [ ] PASS / [ ] FAIL | |
| 13 | Page is usable at mobile width (375px) without horizontal scroll | [ ] PASS / [ ] FAIL | |
| 14 | Page is usable at tablet width (768px) | [ ] PASS / [ ] FAIL | |

## Chrome (mobile emulation, 375px)

| # | Test Case | Result | Notes |
|---|---|---|---|
| 1 | Dashboard loads without console errors | [ ] PASS / [ ] FAIL | |
| 2 | All form inputs (sliders, dropdowns, number input) render correctly | [ ] PASS / [ ] FAIL | |
| 3 | Predict button triggers a request and shows a loading spinner | [ ] PASS / [ ] FAIL | |
| 4 | Risk score cards (3 metrics) display after prediction | [ ] PASS / [ ] FAIL | |
| 5 | Overall risk banner shows correct colour (red/yellow/green) | [ ] PASS / [ ] FAIL | |
| 6 | SHAP bar chart renders with correct red/green bar colours | [ ] PASS / [ ] FAIL | |
| 7 | Equity flag message appears for wealth_quintile=1 inputs | [ ] PASS / [ ] FAIL | |
| 8 | Equity flag message does NOT appear for wealth_quintile=5 inputs | [ ] PASS / [ ] FAIL | |
| 9 | Batch prediction page: CSV upload widget accepts a .csv file | [ ] PASS / [ ] FAIL | |
| 10 | Batch prediction page: results table renders after processing | [ ] PASS / [ ] FAIL | |
| 11 | Batch prediction page: download button produces a valid CSV | [ ] PASS / [ ] FAIL | |
| 12 | About page renders all markdown formatting correctly | [ ] PASS / [ ] FAIL | |
| 13 | Page is usable at mobile width (375px) without horizontal scroll | [ ] PASS / [ ] FAIL | |
| 14 | Page is usable at tablet width (768px) | [ ] PASS / [ ] FAIL | |

## Chrome (tablet emulation, 768px)

| # | Test Case | Result | Notes |
|---|---|---|---|
| 1 | Dashboard loads without console errors | [ ] PASS / [ ] FAIL | |
| 2 | All form inputs (sliders, dropdowns, number input) render correctly | [ ] PASS / [ ] FAIL | |
| 3 | Predict button triggers a request and shows a loading spinner | [ ] PASS / [ ] FAIL | |
| 4 | Risk score cards (3 metrics) display after prediction | [ ] PASS / [ ] FAIL | |
| 5 | Overall risk banner shows correct colour (red/yellow/green) | [ ] PASS / [ ] FAIL | |
| 6 | SHAP bar chart renders with correct red/green bar colours | [ ] PASS / [ ] FAIL | |
| 7 | Equity flag message appears for wealth_quintile=1 inputs | [ ] PASS / [ ] FAIL | |
| 8 | Equity flag message does NOT appear for wealth_quintile=5 inputs | [ ] PASS / [ ] FAIL | |
| 9 | Batch prediction page: CSV upload widget accepts a .csv file | [ ] PASS / [ ] FAIL | |
| 10 | Batch prediction page: results table renders after processing | [ ] PASS / [ ] FAIL | |
| 11 | Batch prediction page: download button produces a valid CSV | [ ] PASS / [ ] FAIL | |
| 12 | About page renders all markdown formatting correctly | [ ] PASS / [ ] FAIL | |
| 13 | Page is usable at mobile width (375px) without horizontal scroll | [ ] PASS / [ ] FAIL | |
| 14 | Page is usable at tablet width (768px) | [ ] PASS / [ ] FAIL | |

## Summary

Total test cases: 70
Passed: ___
Failed: ___

## Issues found (if any)

- 