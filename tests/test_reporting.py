import numpy as np
import pytest

from scripts.generate_model_report import generate_model_report
from src.explainability import SHAPExplainer


def test_normalize_shap_array_handles_binary_output_shape():
    explainer = SHAPExplainer.__new__(SHAPExplainer)
    values = explainer._normalize_shap_array(
        [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6], [0.7, 0.8]])],
        n_rows=2,
        n_cols=2,
    )
    expected = np.array([[0.5, 0.6], [0.7, 0.8]])
    np.testing.assert_array_equal(values, expected)


def test_generate_model_report_includes_shap_feature_summary(tmp_path, monkeypatch):
    report_path = tmp_path / 'model_report.txt'
    monkeypatch.setattr('scripts.generate_model_report.REPORT_PATH', report_path)

    result = generate_model_report()

    assert result is True
    assert report_path.exists()
    content = report_path.read_text(encoding='utf-8')
    assert 'Objective 3 verdict: PASS' in content
    assert 'Stunted — top features:' in content
    assert 'Could not compute SHAP' not in content
