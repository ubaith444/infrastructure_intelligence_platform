"""Milestone 2 Acceptance Test Suite: Incident Prediction.

Verification criteria:
1. 10-minute forward incident emergence risk predictor (P(incident in t + 10m)).
2. Calibrated comparison between Logistic Regression and XGBoost classifiers (PR-AUC, ROC-AUC, Brier score).
3. Evaluation under a fixed 10% operational alert budget quota (Recall at 10% budget).
4. API endpoints for service risk, predictive inference, and evaluation metrics.
"""

from datetime import datetime, timezone
from fastapi.testclient import TestClient
import numpy as np
import pytest

from apps.api.server import app
from data_science.training.train_risk_model import IncidentRiskModel


@pytest.fixture
def client():
    return TestClient(app)


def test_10min_forward_risk_predictor_nominal_and_spiking():
    """Verify 10-minute forward risk predictor distinguishes nominal operations from incident precursors."""
    model = IncidentRiskModel()

    # 1. Nominal feature profile (healthy system)
    nominal_features = {
        "delta_err_1m": 0.001,
        "delta_err_5m": 0.002,
        "delta_err_15m": 0.003,
        "latency_p50": 35.0,
        "latency_p90": 45.0,
        "latency_velocity": 0.1,
        "cpu_pressure": 35.0,
        "memory_pressure": 40.0,
        "request_volume": 120.0,
        "saturation_indicator": 0.40,
        "recent_deployment_event": 0.0,
        "dependency_stress_delta": 0.0,
    }
    nominal_pred = model.predict_risk_details(nominal_features)
    assert nominal_pred["incident_risk_probability"] < 0.25
    assert nominal_pred["risk_band"] in ["LOW", "MODERATE"]
    assert nominal_pred["target_horizon"] == "10_minutes"
    assert nominal_pred["alert_action"] in ["SUPPRESS_ALERT", "DEGRADED_OBSERVABILITY"]

    # 2. Spiking failure precursor profile (incident emergence in 10 minutes)
    incident_features = {
        "delta_err_1m": 0.065,
        "delta_err_5m": 0.110,
        "delta_err_15m": 0.140,
        "latency_p50": 180.0,
        "latency_p90": 340.0,
        "latency_velocity": 5.8,
        "cpu_pressure": 92.0,
        "memory_pressure": 88.0,
        "request_volume": 280.0,
        "saturation_indicator": 0.95,
        "recent_deployment_event": 1.0,
        "dependency_stress_delta": 0.40,
    }
    incident_pred = model.predict_risk_details(incident_features)
    assert incident_pred["incident_risk_probability"] >= 0.70
    assert incident_pred["risk_band"] == "CRITICAL"
    assert incident_pred["alert_action"] == "IMMEDIATE_TRIAGE_ALERT"
    assert len(incident_pred["top_drivers"]) >= 1


def test_calibrated_model_comparison_metrics():
    """Verify calibrated comparison between Logistic Regression and XGBoost classifiers."""
    model = IncidentRiskModel()
    metrics = model.get_evaluation_metrics()

    assert "logistic_regression_baseline" in metrics
    assert "xgboost_classifier" in metrics
    assert "comparison_summary" in metrics
    assert "provenance" in metrics

    lr = metrics["logistic_regression_baseline"]
    xgb = metrics["xgboost_classifier"]

    # 1. Discrimination: Both models should have high ROC-AUC and PR-AUC
    assert lr["roc_auc"] >= 0.80
    assert xgb["roc_auc"] >= 0.85
    assert lr["pr_auc"] >= 0.70
    assert xgb["pr_auc"] >= 0.75

    # 2. Calibration: Brier score loss must be low (closer to 0 is well-calibrated)
    assert lr["brier_score"] <= 0.10
    assert xgb["brier_score"] <= 0.10

    # 3. Transparent provenance reporting
    prov = metrics["provenance"]
    assert prov["training_dataset_source"] == "CONTROLLED_FAULT_INJECTION_SIMULATION"
    assert "10 minutes forward" in prov["target_horizon"]
    assert "critical_limitation" in prov


def test_fixed_10pct_alert_budget_quota_evaluation():
    """Verify recall evaluation under a strict 10% operational alert budget quota."""
    model = IncidentRiskModel()
    metrics = model.get_evaluation_metrics()

    lr_recall = metrics["logistic_regression_baseline"]["recall_at_10pct_budget"]
    xgb_recall = metrics["xgboost_classifier"]["recall_at_10pct_budget"]

    # In a 10% alert quota, the model should capture at least 60% of real impending incidents
    assert lr_recall >= 0.50
    assert xgb_recall >= 0.60
    assert xgb_recall >= lr_recall - 0.05


def test_v1_service_risk_api_endpoint(client):
    """Verify GET /api/v1/services/{id}/risk endpoint."""
    res = client.get("/api/v1/services/order-service/risk")
    assert res.status_code == 200
    data = res.json()
    assert data["service_id"] == "order-service"
    assert "prediction" in data
    assert "features" in data
    assert "incident_risk_probability" in data["prediction"]
    assert "risk_band" in data["prediction"]
    assert "alert_action" in data["prediction"]


def test_v1_risk_predict_endpoint_post(client):
    """Verify POST /api/v1/risk/predict for custom telemetry vector."""
    custom_payload = {
        "features": {
            "delta_err_1m": 0.08,
            "delta_err_5m": 0.12,
            "latency_p90": 380.0,
            "latency_velocity": 6.5,
            "cpu_pressure": 95.0,
            "saturation_indicator": 1.0,
            "recent_deployment_event": 1.0,
        }
    }
    res = client.post("/api/v1/risk/predict", json=custom_payload)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    pred = data["prediction"]
    assert pred["incident_risk_probability"] >= 0.65
    assert pred["risk_band"] in ["HIGH", "CRITICAL"]
    assert pred["alert_action"] in ["EARLY_WARNING_MONITORING", "IMMEDIATE_TRIAGE_ALERT"]


def test_v1_risk_evaluation_endpoint(client):
    """Verify GET /api/v1/risk/evaluation endpoint."""
    res = client.get("/api/v1/risk/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "logistic_regression_baseline" in data
    assert "xgboost_classifier" in data
    assert "comparison_summary" in data
