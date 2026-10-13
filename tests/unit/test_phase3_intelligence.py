from datetime import datetime, timezone
import numpy as np
import pytest

from data_science.evaluation.benchmark_evaluator import BenchmarkEvaluator
from data_science.training.train_risk_model import IncidentRiskModel
from services.anomaly_engine.arbiter import MultiModelAnomalyArbiter
from services.anomaly_engine.isolation_forest import MultivariateIsolationForest
from services.anomaly_engine.statistical_mad import RobustStatisticalDetector
from services.anomaly_engine.streaming_halfspace import RiverStreamingDetector
from services.forecast_engine.forecaster import WorkloadCapacityForecaster
from services.incident_engine.candidate_ranker import CandidateOriginRanker
from services.incident_engine.graph import DependencyGraphEngine
from storage.models import AlertSeverity, AnomalyRecord, MetricWindow


def test_statistical_mad_formula_and_seasonality():
    det = RobustStatisticalDetector(threshold_z=3.5)
    # Feed baseline
    np.random.seed(42)
    for val in np.random.normal(50.0, 2.0, 30):
        det.score("checkout-api", "latency", float(val))

    # Test nominal point
    z_nom, med, is_anom = det.score("checkout-api", "latency", 50.5)
    assert not is_anom
    assert z_nom < 3.5

    # Test extreme outlier
    z_out, _, is_anom_out = det.score("checkout-api", "latency", 300.0)
    assert is_anom_out
    assert z_out > 3.5


def test_zero_mad_fallback_prevents_division_by_zero():
    det = RobustStatisticalDetector()
    # Feed identical values
    for _ in range(25):
        det.score("edge-gateway", "zero_variance_metric", 100.0)

    # Identical value: z should be 0.0 without ZeroDivisionError
    z_score, _, is_anom = det.score("edge-gateway", "zero_variance_metric", 100.0)
    assert z_score == 0.0
    assert not is_anom

    # Small perturbation with zero MAD
    z_dev, _, is_anom_dev = det.score("edge-gateway", "zero_variance_metric", 105.0)
    assert z_dev > 0.0


def test_multivariate_isolation_forest_attribution():
    detector = MultivariateIsolationForest(contamination=0.05, n_estimators=40)
    # Window with high latency and error rate
    w = MetricWindow(
        service_id="payment-service",
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
        request_rate=140.0,
        error_rate=0.08,  # very high error
        latency_p50=180.0,
        latency_p90=450.0,  # extreme latency
        latency_p99=800.0,
        cpu_utilization=85.0,
        memory_utilization=50.0,
    )
    score, is_anom, attrib = detector.score_window(w)
    assert isinstance(score, float)
    assert isinstance(attrib, dict)
    # Attribution weights should sum to approx 1.0
    assert pytest.approx(sum(attrib.values()), 0.05) == 1.0
    assert "latency_p90" in attrib
    assert "error_rate" in attrib


def test_river_streaming_freeze_safeguard():
    detector = RiverStreamingDetector(n_trees=15, height=6, window_size=50)
    w = MetricWindow(
        service_id="inventory-service",
        window_start=datetime.now(timezone.utc),
        window_end=datetime.now(timezone.utc),
        request_rate=100.0,
        error_rate=0.01,
        latency_p50=40.0,
        latency_p90=50.0,
        latency_p99=60.0,
        cpu_utilization=30.0,
        memory_utilization=40.0,
    )
    initial_learned = detector.total_learned
    initial_freeze = detector.freeze_count

    # 1. Normal learning when incident is inactive
    score, is_anom, drift, _ = detector.score_and_learn(w, is_incident_active=False)
    assert detector.total_learned == initial_learned + 1

    # 2. Frozen learning during active incident
    score, is_anom, drift, _ = detector.score_and_learn(w, is_incident_active=True)
    assert detector.freeze_count == initial_freeze + 1


def test_multimodel_arbiter_batch_comparison():
    arb = MultiModelAnomalyArbiter()
    windows = [
        MetricWindow(
            service_id="order-service",
            window_start=datetime.now(timezone.utc),
            window_end=datetime.now(timezone.utc),
            request_rate=100.0 + i,
            error_rate=0.005 if i < 15 else 0.06,  # injected failure late
            latency_p50=40.0,
            latency_p90=45.0 if i < 15 else 250.0,
            latency_p99=60.0,
            cpu_utilization=30.0 if i < 15 else 90.0,
            memory_utilization=40.0,
        )
        for i in range(25)
    ]
    comparison = arb.compare_on_batch(windows, active_incident=False)
    assert comparison["total_windows_evaluated"] == 25
    assert "statistical_mad_anomaly_rate" in comparison
    assert "isolation_forest_anomaly_rate" in comparison
    assert "river_streaming_anomaly_rate" in comparison


def test_incident_risk_model_training_and_budget_recall():
    risk_mod = IncidentRiskModel()
    metrics = risk_mod.get_evaluation_metrics()

    assert "logistic_regression_baseline" in metrics
    assert "xgboost_classifier" in metrics
    assert "provenance" in metrics

    lr_res = metrics["logistic_regression_baseline"]
    xgb_res = metrics["xgboost_classifier"]

    # Both models should exhibit strong discrimination
    assert lr_res["roc_auc"] > 0.70
    assert xgb_res["roc_auc"] > 0.75
    # XGBoost should match or outperform baseline on PR-AUC
    assert xgb_res["pr_auc"] >= lr_res["pr_auc"] - 0.05
    # Recall at 10% budget should be tracked
    assert "recall_at_10pct_budget" in xgb_res

    # Test prediction details
    pred = risk_mod.predict_risk_details({
        "delta_err_1m": 0.05,
        "delta_err_5m": 0.08,
        "latency_p90": 300.0,
        "latency_velocity": 4.5,
        "cpu_pressure": 85.0,
        "saturation_indicator": 0.95,
        "recent_deployment_event": 1.0,
    })
    assert pred["risk_band"] in ["CRITICAL", "HIGH"]
    assert "alert_action" in pred
    assert len(pred["top_drivers"]) > 0


def test_capacity_forecaster_multi_horizon_intervals():
    forecaster = WorkloadCapacityForecaster()
    forecast = forecaster.predict_capacity(current_val=150.0, capacity_limit=250.0, metric="request_rate")

    assert forecast["current_metric_value"] == 150.0
    assert "forecasts" in forecast

    # Check 5m, 15m, 60m horizons
    for h in [5, 15, 60]:
        h_key = f"horizon_{h}m"
        assert h_key in forecast["forecasts"]
        f_data = forecast["forecasts"][h_key]
        assert f_data["interval_p10"] <= f_data["tree_median_forecast"]
        assert f_data["tree_median_forecast"] <= f_data["interval_p90"]
        assert "seasonal_naive_baseline" in f_data
        assert f_data["status"] in ["NOMINAL", "EARLY_WARNING", "CRITICAL_SATURATION"]

    # Check comparative evaluation benchmarks
    benchmarks = forecaster.get_evaluation_benchmarks()
    assert "horizon_15m" in benchmarks
    assert "mae" in benchmarks["horizon_15m"]["tree_quantile_model"]
    assert "interval_coverage_p10_p90" in benchmarks["horizon_15m"]["tree_quantile_model"]


def test_candidate_origin_ranker_error_propagation_evidence():
    graph = DependencyGraphEngine()
    ranker = CandidateOriginRanker(graph)

    t0 = datetime(2026, 10, 13, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 13, 10, 0, 30, tzinfo=timezone.utc)

    # payment-service error anomaly at t0
    anom_payment = AnomalyRecord(
        anomaly_id="a1",
        service_id="payment-service",
        detector_type="RobustStatisticalMAD",
        timestamp=t0,
        severity=AlertSeverity.CRITICAL,
        score=5.5,
        threshold=3.5,
        metric_attributions={"error_rate": 0.8},
        narrative="Statistical error rate deviation on payment-service",
    )
    # checkout-api latency anomaly at t1
    anom_checkout = AnomalyRecord(
        anomaly_id="a2",
        service_id="checkout-api",
        detector_type="RobustStatisticalMAD",
        timestamp=t1,
        severity=AlertSeverity.WARNING,
        score=4.0,
        threshold=3.5,
        metric_attributions={"latency_p90": 0.7},
        narrative="Statistical latency deviation on checkout-api",
    )

    candidates = ranker.rank_candidates([anom_payment, anom_checkout])
    assert len(candidates) >= 2
    # payment-service should rank highest due to earlier time and upstream dependency
    top = candidates[0]
    assert top["service_id"] == "payment-service"
    assert any("checkout-api latency increased after payment-service errors rose" in ev for ev in top["propagation_evidence"])
    # Check associative wording
    assert "observational telemetry does not assert unverified causality" in top["associative_narrative"]


def test_benchmark_evaluator_smd_suite():
    evaluator = BenchmarkEvaluator()
    results = evaluator.run_smd_benchmark(n_points=250)

    assert "detectors" in results
    assert "robust_statistical_mad" in results["detectors"]
    assert "multivariate_isolation_forest" in results["detectors"]
    assert "river_streaming_halfspace" in results["detectors"]

    mad_res = results["detectors"]["robust_statistical_mad"]
    assert "point_adjusted_f1" in mad_res
    assert "mean_inference_latency_us" in mad_res
    assert mad_res["point_adjusted_f1"] >= 0.70
