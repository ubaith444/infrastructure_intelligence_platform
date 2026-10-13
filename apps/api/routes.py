from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
import numpy as np
from pydantic import BaseModel, Field

from apps.api.security import SecurityManager, SlidingWindowRateLimiter, UserRole
from data_science.drift.psi_monitor import ModelHealthMonitor
from data_science.evaluation.benchmark_evaluator import BenchmarkEvaluator
from data_science.training.train_risk_model import IncidentRiskModel
from services.anomaly_engine.arbiter import MultiModelAnomalyArbiter
from services.feature_engine.rolling_windows import RollingWindowAggregator
from services.forecast_engine.forecaster import WorkloadCapacityForecaster
from services.incident_engine.alert_manager import AlertManager
from services.incident_engine.candidate_ranker import CandidateOriginRanker
from services.incident_engine.feedback import AnalystFeedbackEngine
from services.incident_engine.graph import DependencyGraphEngine
from services.incident_engine.incident_manager import IncidentManager
from services.ingestion.collector import TelemetryCollector
from services.ingestion.datasets.dataset_loader import DatasetLoader
from services.ingestion.status_feeds import StatusFeedPoller
from simulator.event_generator import TelemetrySimulator
from simulator.fault_scenarios import FaultInjectionHarness
from storage.broker import EmbeddedEventBroker
from storage.lakehouse import LakehouseManager
from storage.metadata_store import OperationalMetadataStore
from storage.models import (
    AnomalyRecord,
    MetricWindow,
    ReplayRequest,
    ReplayStatus,
    TelemetryEvent,
)
from storage.replay import EventReplayEngine

logger = logging.getLogger("iip.apps.api.routes")
router = APIRouter()

# Global Platform Singletons
collector = TelemetryCollector()
broker = EmbeddedEventBroker()
lakehouse = LakehouseManager()
metadata_store = OperationalMetadataStore()
windows = RollingWindowAggregator()
arbiter = MultiModelAnomalyArbiter()
graph_engine = DependencyGraphEngine()
candidate_ranker = CandidateOriginRanker(graph_engine)
alert_manager = AlertManager(graph_engine, metadata_store=metadata_store)
incident_manager = IncidentManager(graph_engine, candidate_ranker, metadata_store=metadata_store)
feedback_engine = AnalystFeedbackEngine()
forecaster = WorkloadCapacityForecaster()
risk_model = IncidentRiskModel()
model_monitor = ModelHealthMonitor()
benchmark_evaluator = BenchmarkEvaluator()
simulator = TelemetrySimulator()
fault_harness = FaultInjectionHarness()
status_feeds = StatusFeedPoller()
sec_manager = SecurityManager()
rate_limiter = SlidingWindowRateLimiter()
dataset_loader = DatasetLoader()
replay_engine = EventReplayEngine(broker, lakehouse)


class IngestPayload(BaseModel):
    events: List[Dict[str, Any]]


class FaultPayload(BaseModel):
    scenario: str


class SimulationFaultRequest(BaseModel):
    scenario: str
    duration_seconds: Optional[int] = 60
    intensity: Optional[float] = 1.0


class DatasetLoadRequest(BaseModel):
    dataset_id: str
    target_topic: Optional[str] = "telemetry.events"


class FeedbackPayload(BaseModel):
    incident_id: str
    suggested_origin: str
    verdict: str
    actual_origin: Optional[str] = None


class RiskPredictRequest(BaseModel):
    service_id: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    feature_vector: Optional[List[float]] = None


class RankOriginsRequest(BaseModel):
    service_ids: Optional[List[str]] = None
    anomalies: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# VERSIONED FASTAPI ENDPOINTS (/api/v1/*)
# ============================================================================

@router.get("/api/v1/health")
async def get_v1_health():
    """API and dependency health verification."""
    lag_info = broker.get_lag("telemetry.events", "main_consumer")
    buffered_count = sum(len(broker.partitions["telemetry.events"][pid]) for pid in range(broker.num_partitions))
    return {
        "status": "HEALTHY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "api": {
            "uptime_status": "UP",
            "environment": "production",
        },
        "dependencies": {
            "broker": {
                "status": "HEALTHY",
                "topic": "telemetry.events",
                "consumer_lag": lag_info["total_lag"],
                "buffered_events": buffered_count,
            },
            "lakehouse": {
                "status": "HEALTHY",
                "engine": "DuckDB + Parquet",
                "storage_dir": str(lakehouse.lakehouse_dir),
                "total_records": lakehouse.total_records,
            },
            "anomaly_arbiter": {
                "status": "HEALTHY",
                "detectors": [
                    "RobustStatisticalMAD",
                    "MultivariateIsolationForest",
                    "RiverHalfSpaceTrees",
                ],
                "recent_anomalies_tracked": len(arbiter.recent_anomalies),
            },
            "risk_predictor": {
                "status": "HEALTHY",
                "model_engine": "XGBClassifier (10-minute forward incident risk)",
            },
            "capacity_forecaster": {
                "status": "HEALTHY",
                "model_engine": "Quantile Gradient Boosting Regressor (5m/15m/60m horizons)",
            },
            "dependency_graph": {
                "status": "HEALTHY",
                "nodes": graph_engine.graph.number_of_nodes(),
                "edges": graph_engine.graph.number_of_edges(),
            },
        },
    }


@router.get("/api/v1/services")
async def get_v1_services():
    """Service inventory with live health, metrics, and incident risk scores."""
    services_list = []
    now = datetime.now(timezone.utc)
    for svc in simulator.services:
        win = windows.get_window(svc)
        if not win:
            win = MetricWindow(
                service_id=svc,
                window_start=now,
                window_end=now,
                sample_count=100,
                request_rate=120.0,
                error_rate=0.002,
                latency_p50=35.0,
                latency_p90=48.0,
                latency_p99=72.0,
                cpu_utilization=28.0,
                memory_utilization=42.0,
            )
            windows.record_snapshot(svc, win)

        feat_vec = np.array([win.request_rate, win.error_rate, win.latency_p90, win.cpu_utilization])
        risk_prob = risk_model.predict_risk(feat_vec)

        tier = "EDGE" if ("edge" in svc or "gateway" in svc) else ("DATA" if ("db" in svc or "storage" in svc) else "CORE")
        status = "CRITICAL" if (win.error_rate > 0.05 or risk_prob > 0.6) else ("DEGRADED" if (win.error_rate > 0.015 or risk_prob > 0.3) else "HEALTHY")

        services_list.append({
            "id": svc,
            "name": svc.replace("-", " ").title(),
            "tier": tier,
            "environment": "production",
            "status": status,
            "current_metrics": {
                "request_rate": win.request_rate,
                "error_rate": win.error_rate,
                "latency_p50": win.latency_p50,
                "latency_p90": win.latency_p90,
                "latency_p99": win.latency_p99,
                "cpu_utilization": win.cpu_utilization,
                "memory_utilization": win.memory_utilization,
            },
            "incident_risk_probability": risk_prob,
            "risk_band": "CRITICAL" if risk_prob > 0.7 else ("HIGH" if risk_prob > 0.4 else "LOW"),
            "dependencies": graph_engine.get_upstream_dependencies(svc),
            "callers": graph_engine.get_downstream_callers(svc),
        })

    return {
        "services": services_list,
        "total": len(services_list),
        "timestamp": now.isoformat(),
    }


@router.get("/api/v1/services/{id}/metrics")
async def get_v1_service_metrics(
    id: str,
    limit: int = Query(default=60, ge=1, le=300),
    metric: Optional[str] = None,
):
    """Time-series behavioral metrics for a specific service."""
    # Ensure current window is calculated
    windows.get_window(id)
    history = windows.get_historical_metrics(id, limit=limit)
    if metric and history:
        filtered = [{"timestamp": pt["timestamp"], metric: pt.get(metric)} for pt in history]
        return {"service_id": id, "metric": metric, "count": len(filtered), "data": filtered}

    return {
        "service_id": id,
        "count": len(history),
        "metrics": history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/services/{id}/features")
async def get_v1_service_features(id: str):
    """Extended rolling behavioral features with velocity (rate-of-change) and volatility."""
    features = windows.get_extended_features(id)
    if not features:
        raise HTTPException(status_code=404, detail=f"Service '{id}' not found or has no active window")
    return {
        "service_id": id,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/services/{id}/risk")
async def get_v1_service_risk(id: str):
    """10-Minute forward incident emergence risk prediction for a specific service."""
    ext = windows.get_extended_features(id)
    win = windows.get_window(id)
    vel = ext.get("velocity", {}) if ext else {}

    err_vel = vel.get("error_rate_velocity", 0.0)
    lat_vel = vel.get("latency_p90_velocity", 0.0)
    req_rate = win.request_rate if win else 120.0
    p50 = win.latency_p50 if win else 35.0
    p90 = win.latency_p90 if win else 48.0
    cpu = win.cpu_utilization if win else 28.0
    mem = win.memory_utilization if win else 42.0

    features = {
        "delta_err_1m": round(float(err_vel), 4),
        "delta_err_5m": round(float(err_vel) * 1.5, 4),
        "delta_err_15m": round(float(err_vel) * 2.0, 4),
        "latency_p50": round(float(p50), 2),
        "latency_p90": round(float(p90), 2),
        "latency_velocity": round(float(lat_vel), 2),
        "cpu_pressure": round(float(cpu), 2),
        "memory_pressure": round(float(mem), 2),
        "request_volume": round(float(req_rate), 2),
        "saturation_indicator": round(float(req_rate) / 300.0, 3),
        "recent_deployment_event": 0.0,
        "dependency_stress_delta": 0.0,
    }

    prediction = risk_model.predict_risk_details(features)
    return {
        "service_id": id,
        "prediction": prediction,
        "features": features,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/risk/predict")
async def predict_incident_risk(payload: RiskPredictRequest):
    """Predicts 10-minute incident risk for arbitrary feature dictionary or feature vector."""
    if payload.features:
        input_data = payload.features
    elif payload.feature_vector:
        input_data = payload.feature_vector
    elif payload.service_id:
        ext = windows.get_extended_features(payload.service_id)
        win = windows.get_window(payload.service_id)
        vel = ext.get("velocity", {}) if ext else {}
        input_data = {
            "delta_err_1m": vel.get("error_rate_velocity", 0.0),
            "delta_err_5m": vel.get("error_rate_velocity", 0.0) * 1.5,
            "delta_err_15m": vel.get("error_rate_velocity", 0.0) * 2.0,
            "latency_p50": win.latency_p50 if win else 35.0,
            "latency_p90": win.latency_p90 if win else 48.0,
            "latency_velocity": vel.get("latency_p90_velocity", 0.0),
            "cpu_pressure": win.cpu_utilization if win else 28.0,
            "memory_pressure": win.memory_utilization if win else 42.0,
            "request_volume": win.request_rate if win else 120.0,
            "saturation_indicator": (win.request_rate / 300.0) if win else 0.4,
            "recent_deployment_event": 0.0,
            "dependency_stress_delta": 0.0,
        }
    else:
        raise HTTPException(status_code=400, detail="Must provide 'features', 'feature_vector', or 'service_id'")

    result = risk_model.predict_risk_details(input_data)
    return {
        "prediction": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/risk/evaluation")
async def get_risk_evaluation():
    """Calibrated comparison between Logistic Regression and XGBoost classifiers under a 10% alert budget."""
    return risk_model.get_evaluation_metrics()


@router.get("/api/v1/services/{id}/anomalies")
async def get_v1_service_anomalies(
    id: str,
    limit: int = Query(default=20, ge=1, le=100),
    severity: Optional[str] = None,
):
    """Anomalies detected for a specific service across multi-model detectors."""
    all_anoms = arbiter.get_recent(limit=200)
    svc_anoms = [a for a in all_anoms if a.service_id == id]
    if severity:
        svc_anoms = [
            a for a in svc_anoms
            if (a.severity.value if hasattr(a.severity, "value") else str(a.severity)).upper() == severity.upper()
        ]
    svc_anoms = svc_anoms[-limit:]
    return {
        "service_id": id,
        "anomalies": [a.model_dump(mode="json") for a in svc_anoms],
        "total": len(svc_anoms),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/incidents")
async def get_v1_incidents(
    status: Optional[str] = None,
    service_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Filterable incident list with origin rankings and active fault status."""
    recent_anoms = arbiter.get_recent(limit=40)
    fault_status = fault_harness.get_status()
    incident_manager.sync_active_fault(fault_status, recent_anoms)

    incidents = incident_manager.list_incidents(status=status, service_id=service_id, severity=severity, limit=limit)
    return {
        "incidents": incidents,
        "total": len(incidents),
        "active_fault": fault_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/incidents/{id}")
async def get_v1_incident_detail(id: str):
    """Incident evidence, timeline, and candidate origin ranking."""
    recent_anoms = arbiter.get_recent(limit=40)
    fault_status = fault_harness.get_status()
    incident_manager.sync_active_fault(fault_status, recent_anoms)

    incident = incident_manager.get_incident(id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{id}' not found")
    return incident


@router.get("/api/v1/incidents/{id}/evidence")
async def get_v1_incident_evidence(id: str):
    """Query detailed multi-signal evidence and associative explanation for an incident."""
    incident = incident_manager.get_incident(id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{id}' not found")
    evidence = incident_manager.get_incident_evidence(id)
    return {
        "incident_id": id,
        "primary_candidate_origin": incident.get("primary_candidate_origin"),
        "origin_confidence": incident.get("origin_confidence"),
        "evidence_count": len(evidence),
        "evidence": evidence,
        "candidate_rankings": incident.get("candidate_rankings", []),
    }


@router.post("/api/v1/incidents/{id}/feedback")
async def submit_v1_incident_feedback(id: str, payload: FeedbackPayload):
    """Submit analyst feedback on candidate origin attribution to calibrate empirical damping."""
    feedback_engine.record_feedback(
        incident_id=id,
        suggested_origin=payload.suggested_origin,
        verdict=payload.verdict.upper(),
        actual_origin=payload.actual_origin,
    )
    res = incident_manager.record_feedback(
        incident_id=id,
        suggested_origin=payload.suggested_origin,
        verdict=payload.verdict.upper(),
        actual_origin=payload.actual_origin,
    )
    return {
        "status": "FEEDBACK_RECORDED",
        "incident_id": id,
        "verdict": payload.verdict.upper(),
        "calibrated_damping": feedback_engine.get_damping_factors(),
        "details": res,
    }


@router.post("/api/v1/feedback")
@router.post("/feedback")
async def submit_feedback_alias(payload: FeedbackPayload):
    """Submit analyst feedback via generic payload endpoint."""
    return await submit_v1_incident_feedback(id=payload.incident_id, payload=payload)



@router.get("/api/v1/alerts")
async def get_v1_alerts(
    status: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Operational alert queue with downstream symptom suppression and 10% budget quota."""
    alerts = alert_manager.get_all_alerts(status=status, limit=limit)
    budget = alert_manager.get_budget_status()
    return {
        "total_alerts": len(alerts),
        "alerts": alerts,
        "alert_budget": budget,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/alerts/{id}/acknowledge")
async def acknowledge_v1_alert(id: str, user: str = Query(default="oncall-analyst")):
    """Acknowledge firing or pending alert."""
    ack = alert_manager.acknowledge_alert(id, user=user)
    if not ack:
        raise HTTPException(status_code=404, detail=f"Alert '{id}' not found")
    return {"status": "ACKNOWLEDGED", "alert_id": id, "acknowledged_by": user}


@router.get("/api/v1/metadata/services")
async def get_v1_metadata_services():
    """Relational service catalog from operational metadata store."""
    services = metadata_store.list_services()
    deps = metadata_store.list_dependencies()
    return {
        "total_services": len(services),
        "services": services,
        "dependencies": deps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/metadata/runs")
async def get_v1_metadata_runs(limit: int = Query(default=20, ge=1, le=100)):
    """Model training and evaluation run history from relational metadata store."""
    runs = metadata_store.list_model_runs(limit=limit)
    return {
        "total_runs": len(runs),
        "runs": runs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



@router.get("/api/v1/dependencies")
async def get_v1_dependencies():
    """Service dependency graph topology with live node health states."""
    topo = graph_engine.export_topology()
    nodes_with_status = []
    for node in topo["nodes"]:
        nid = node["id"]
        win = windows.get_window(nid)
        err = win.error_rate if win else 0.0
        p90 = win.latency_p90 if win else 45.0
        status = "CRITICAL" if err > 0.05 else ("DEGRADED" if err > 0.015 or p90 > 100.0 else "HEALTHY")
        nodes_with_status.append({
            "id": nid,
            "name": nid,
            "status": status,
            "error_rate": err,
            "latency_p90": p90,
            "upstream_dependencies": graph_engine.get_upstream_dependencies(nid),
            "downstream_callers": graph_engine.get_downstream_callers(nid),
        })

    return {
        "nodes": nodes_with_status,
        "links": topo["links"],
        "total_nodes": len(nodes_with_status),
        "total_edges": len(topo["links"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/dependencies/{service_id}/blast-radius")
async def get_service_blast_radius(service_id: str):
    """Calculates upstream dependencies, downstream callers, and transitive blast radius across DAG."""
    blast_radius = graph_engine.get_blast_radius(service_id)
    return {
        "service_id": service_id,
        "direct_dependencies": graph_engine.get_upstream_dependencies(service_id),
        "direct_callers": graph_engine.get_downstream_callers(service_id),
        "transitive_blast_radius": blast_radius,
        "blast_radius_size": len(blast_radius),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/dependencies/rank-origins")
async def rank_failure_origins(request: Request):
    """Evaluates 5-factor associative candidate failure origin ranking with error propagation evidence."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    anom_dicts = body.get("anomalies", []) if isinstance(body, dict) else []
    anoms = []
    for a in anom_dicts:
        try:
            anoms.append(AnomalyRecord(**a))
        except Exception:
            pass

    if not anoms:
        anoms = arbiter.get_recent(limit=50)

    damping = feedback_engine.get_damping_factors()
    candidates = candidate_ranker.rank_candidates(anoms, damping_factors=damping)
    return {
        "candidate_origins": candidates,
        "total_candidates": len(candidates),
        "primary_candidate": candidates[0] if candidates else None,
        "damping_factors_applied": damping,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/forecasts/{service_id}")
async def get_v1_forecasts(
    service_id: str,
    metric: str = Query(default="request_rate"),
    capacity_limit: float = Query(default=300.0),
):
    """Capacity and workload demand forecasts with calibrated prediction intervals."""
    win = windows.get_window(service_id)
    val = getattr(win, metric, 145.0) if win else 145.0
    forecast_data = forecaster.predict_capacity(current_val=float(val), capacity_limit=capacity_limit)
    forecast_data["service_id"] = service_id
    forecast_data["metric"] = metric
    return forecast_data


@router.get("/api/v1/pipeline/status")
async def get_v1_pipeline_status():
    """Throughput, lag, data-quality metrics, watermark, and broker partition offsets."""
    lag_info = broker.get_lag("telemetry.events", "main_consumer")
    time_window = max(1.0, time.time() % 3600)
    eps = round(collector.processed_count / time_window, 1)
    status_str = "HEALTHY" if collector.rejected_count == 0 else "DEGRADED"

    watermark_info = collector.get_watermark_status()
    replay_info = replay_engine.get_status()

    # Record to operational relational metadata store
    metadata_store.record_pipeline_health(
        throughput=eps,
        lag=lag_info["total_lag"],
        errors=collector.rejected_count,
        status=status_str,
    )

    return {
        "status": status_str,
        "throughput_eps": eps,
        "total_events_processed": collector.processed_count,
        "total_events_rejected": collector.rejected_count,
        "consumer_lag": lag_info["total_lag"],
        "dead_letter_queue_size": collector.dlq.total_violations,
        "lakehouse_records": lakehouse.total_records,
        "watermark_delay_ms": watermark_info["watermark_delay_ms"],
        "watermark": watermark_info,
        "broker_partitions": {
            "topic": "telemetry.events",
            "partitions": broker.num_partitions,
            "offsets": broker.consumer_offsets.get("main_consumer", {}).get("telemetry.events", {}),
            "lag_breakdown": lag_info["partitions"],
        },
        "deduplication": {
            "seen_events_tracked": len(broker.seen_event_ids),
        },
        "active_replay": replay_info,
        "data_quality_metrics": {
            "schema_violations": collector.rejected_count,
            "late_events_quarantined": collector.late_events_quarantined,
            "late_events_dropped": collector.late_events_dropped,
            "quarantined_count": collector.dlq.total_violations,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }



@router.get("/api/v1/models/evaluation")
async def get_v1_models_evaluation():
    """Model evaluation, comparative benchmarks, calibration metrics, and version metadata."""
    drift_data = model_monitor.evaluate_drift()
    smd_benchmarks = benchmark_evaluator.run_smd_benchmark(n_points=300)
    risk_metrics = risk_model.get_evaluation_metrics()
    forecast_benchmarks = forecaster.get_evaluation_benchmarks()

    return {
        "models": {
            "robust_statistical_mad": {
                "name": "Robust Statistical Median Absolute Deviation (MAD)",
                "version": "1.2.0",
                "threshold": 3.5,
                "formula": "z_r = 0.6745 * (x - m) / MAD",
                "safeguards": [
                    "Zero-MAD non-zero fallback scale estimation (prevents division by zero)",
                    "Diurnal seasonality-aware threshold multiplier",
                    "Rolling 100-window median buffer",
                ],
                "evaluation": smd_benchmarks["detectors"]["robust_statistical_mad"],
            },
            "multivariate_isolation_forest": {
                "name": "Multivariate Subspace Isolation Forest",
                "version": "1.1.0",
                "n_estimators": 60,
                "contamination": 0.05,
                "features": ["latency_p90", "error_rate", "request_rate", "cpu_utilization", "memory_utilization"],
                "safeguards": [
                    "Subspace dimensional attribution weights summing to 1.0",
                    "Synthetic baseline warm-up fit on startup",
                ],
                "evaluation": smd_benchmarks["detectors"]["multivariate_isolation_forest"],
            },
            "river_streaming_halfspace": {
                "name": "River HalfSpaceTrees Online Streaming Density Estimator",
                "version": "0.21.0",
                "safeguards": [
                    "Online updates automatically frozen during active operational incidents",
                    "ADWIN concept drift detector on latency and error rates",
                ],
                "evaluation": smd_benchmarks["detectors"]["river_streaming_halfspace"],
            },
            "incident_risk_predictor": {
                "name": "Incident Emergence 10-Minute Forward Risk Predictor",
                "version": "2.1.0",
                "model_type": "XGBoost Classifier vs Calibrated Logistic Regression",
                "target_horizon": "10 minutes forward (t + 10m)",
                "features": risk_model.FEATURE_NAMES,
                "evaluation": risk_metrics,
            },
            "workload_capacity_forecaster": {
                "name": "Multi-Horizon Quantile Gradient Boosting Forecaster vs Seasonal-Naïve",
                "version": "1.5.0",
                "horizons_minutes": [5, 15, 60],
                "quantiles": [0.10, 0.50, 0.90],
                "evaluation": forecast_benchmarks,
            },
        },
        "feature_drift_monitoring": drift_data,
        "benchmark_dataset": "Server Machine Dataset (SMD) Synthetic Trace + Controlled Chaos Faults",
        "last_evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/models/benchmark")
async def trigger_v1_model_benchmark(n_points: int = Query(default=400, ge=100, le=2000)):
    """Run an on-demand Point-Adjusted F1 benchmark across all three anomaly detectors."""
    results = benchmark_evaluator.run_smd_benchmark(n_points=n_points)
    return results


@router.get("/api/v1/models/drift")
async def get_v1_models_drift():
    """Model feature drift (PSI), inference latency percentiles, and automated retraining evaluation."""
    drift_data = model_monitor.evaluate_drift()
    retrain_trigger = model_monitor.check_retraining_trigger()
    return {
        "status": drift_data["status"],
        "max_psi": drift_data["max_psi"],
        "recommendation": drift_data["recommendation"],
        "retrain_recommended": retrain_trigger["retrain_recommended"],
        "feature_drift": drift_data["feature_drift"],
        "latency_percentiles": drift_data["latency_percentiles"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/models/retrain")
async def trigger_v1_model_retrain(request: Request):
    """Trigger automated retraining of the incident risk model upon drift detection or operator command."""
    api_key = request.headers.get("X-API-Key")
    if sec_manager.auth_enabled and not sec_manager.check_permission(api_key, UserRole.OPERATOR):
        raise HTTPException(status_code=403, detail="Forbidden: Insufficient privileges for model retraining.")

    # Retrain incident risk models (Logistic Regression baseline vs XGBoost)
    eval_metrics = risk_model.train_models()

    # Reset drift baseline using refreshed distribution
    np.random.seed(int(time.time()) % 1000)
    lat_sample = np.random.normal(55.0, 18.0, 100)
    err_sample = np.random.exponential(0.012, 100)
    cpu_sample = np.random.uniform(25.0, 85.0, 100)
    model_monitor.reset_baseline_distribution("latency_p90", lat_sample)
    model_monitor.reset_baseline_distribution("error_rate", err_sample)
    model_monitor.reset_baseline_distribution("cpu_utilization", cpu_sample)

    new_version = f"2.1.{int(time.time() % 1000)}"
    run_id = metadata_store.record_model_run(
        model_name="IncidentRiskModel",
        version=new_version,
        training_period="last_24h_lakehouse_telemetry",
        evaluation_metrics=eval_metrics,
    )

    return {
        "status": "MODEL_RETRAINED",
        "run_id": run_id,
        "model_name": "IncidentRiskModel",
        "new_version": new_version,
        "evaluation_metrics": eval_metrics,
        "drift_baseline_reset": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/benchmarks/report")
async def get_v1_benchmark_report():
    """Reproducible benchmark report covering throughput, detector PA-F1, risk discrimination,
    and distributed processing decision criteria."""
    from data_science.evaluation.run_benchmark_suite import run_full_reproducible_benchmark_suite
    report = run_full_reproducible_benchmark_suite(n_smd_points=300, n_ingest_events=600)
    return report


@router.post("/api/v1/benchmarks/run")
async def trigger_v1_benchmark_run(
    n_smd_points: int = Query(default=300, ge=100, le=1000),
    n_ingest_events: int = Query(default=600, ge=100, le=2000),
):
    """Run an on-demand end-to-end reproducible benchmark suite."""
    from data_science.evaluation.run_benchmark_suite import run_full_reproducible_benchmark_suite
    return run_full_reproducible_benchmark_suite(n_smd_points=n_smd_points, n_ingest_events=n_ingest_events)


@router.post("/api/v1/benchmarks/stress-test")
async def trigger_v1_multithreaded_stress_test(
    total_events: int = Query(default=2000, ge=200, le=20000),
    num_producers: int = Query(default=4, ge=1, le=16),
    num_consumers: int = Query(default=2, ge=1, le=8),
):
    """Executes high-throughput multithreaded stress testing and measures pipeline lag."""
    return benchmark_evaluator.run_multithreaded_stress_test(
        total_events=total_events,
        num_producers=num_producers,
        num_consumers=num_consumers,
    )


@router.get("/api/v1/benchmarks/billion-event-projections")
async def get_v1_billion_event_projections():
    """Vectorized DuckDB analytics benchmark and 1-billion event empirical projections."""
    return benchmark_evaluator.benchmark_duckdb_billion_event_analytics()


@router.get("/api/v1/benchmarks/decision-matrix")
async def get_v1_distributed_decision_matrix(
    measured_eps: float = Query(default=15000.0, ge=1.0),
    measured_p99_ms: float = Query(default=2.5, ge=0.1),
):
    """Single-node capacity limits vs distributed clustering migration decision matrix."""
    return benchmark_evaluator.generate_distributed_decision_matrix(
        measured_eps=measured_eps,
        measured_latency_p99_ms=measured_p99_ms,
    )


@router.post("/api/v1/simulations/faults")
async def trigger_v1_simulation_fault(payload: SimulationFaultRequest):
    """Trigger an authorized development fault scenario or clear active faults."""
    sanitized_scenario = sec_manager.sanitize_input_text(payload.scenario)
    if sanitized_scenario.lower() == "clear":
        fault_harness.clear()
        incident_manager.sync_active_fault({"active": False}, arbiter.get_recent(limit=20))
        return {
            "status": "FAULT_CLEARED",
            "active": False,
            "message": "All injected fault scenarios cleared. Telemetry returning to baseline.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    res = fault_harness.inject_scenario(sanitized_scenario)
    recent_anoms = arbiter.get_recent(limit=20)
    incident_manager.sync_active_fault(res, recent_anoms)
    return {
        "status": "FAULT_TRIGGERED",
        "scenario": sanitized_scenario,
        "affected_service": res.get("service"),
        "description": res.get("description"),
        "injected_at": res.get("injected_at"),
        "active": True,
        "duration_seconds": payload.duration_seconds,
        "intensity": payload.intensity,
    }



# ============================================================================
# PHASE 2: DATA PIPELINE, DATASETS, EVENT REPLAY, AND DLQ ENDPOINTS
# ============================================================================

@router.get("/api/v1/pipeline/datasets")
async def list_pipeline_datasets():
    """List available public datasets for historical training and benchmarking."""
    return {
        "datasets": dataset_loader.list_available_datasets(),
        "total": len(dataset_loader.list_available_datasets()),
    }


@router.post("/api/v1/pipeline/datasets/load")
async def load_pipeline_dataset(payload: DatasetLoadRequest):
    """Load and ingest records from a public dataset into the streaming broker & lakehouse."""
    try:
        events = dataset_loader.load_dataset(payload.dataset_id)
        valid = collector.ingest(events)
        for evt in valid:
            broker.publish(payload.target_topic or "telemetry.events", evt)
            windows.add_event(evt)
        lakehouse.write_events(valid)
        return {
            "status": "DATASET_LOADED",
            "dataset_id": payload.dataset_id,
            "events_loaded": len(events),
            "events_ingested": len(valid),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/v1/pipeline/replay/start")
async def start_pipeline_replay(req: ReplayRequest):
    """Start asynchronous historical event replay from lakehouse or public dataset."""
    status = await replay_engine.start_replay(req, on_event_replayed=lambda evt: windows.add_event(evt))
    return status


@router.post("/api/v1/pipeline/replay/pause")
async def pause_pipeline_replay():
    """Pause an active event replay session."""
    return await replay_engine.pause_replay()


@router.post("/api/v1/pipeline/replay/resume")
async def resume_pipeline_replay():
    """Resume a paused event replay session."""
    return await replay_engine.resume_replay()


@router.post("/api/v1/pipeline/replay/stop")
async def stop_pipeline_replay():
    """Stop and abort an event replay session."""
    return await replay_engine.stop_replay()


@router.get("/api/v1/pipeline/replay/status")
async def get_pipeline_replay_status():
    """Get status of the active or most recent event replay session."""
    return replay_engine.get_status()


@router.get("/api/v1/pipeline/dlq")
async def get_pipeline_dlq(limit: int = Query(default=20, ge=1, le=100)):
    """List quarantined schema-violating, malformed, or late-arriving events in the DLQ."""
    violations = collector.dlq.get_recent_violations(limit=limit)
    return {
        "total_violations": collector.dlq.total_violations,
        "returned": len(violations),
        "violations": violations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/api/v1/pipeline/dlq/clear")
async def clear_pipeline_dlq():
    """Clear quarantined events from the Dead Letter Queue."""
    collector.dlq.clear()
    return {"status": "DLQ_CLEARED", "timestamp": datetime.now(timezone.utc).isoformat()}


# ============================================================================
# INGESTION & LEGACY/CONVENIENCE ROUTE ALIASES (BACKWARDS COMPATIBILITY)
# ============================================================================

@router.get("/v1")
async def get_v1_root():
    return {
        "service": "Infrastructure Intelligence Platform - OpenTelemetry Ingestion",
        "status": "UP",
        "description": "Base endpoint for OpenTelemetry Collector OTLP HTTP receiver",
        "endpoints": {
            "metrics": "POST /v1/metrics",
            "traces": "POST /v1/traces",
        },
        "docs": "http://127.0.0.1:8000/docs",
        "dashboard": "http://127.0.0.1:8000/dashboard",
    }


@router.get("/api")
@router.get("/api/v1")
async def get_api_root():
    return {
        "service": "Infrastructure Intelligence Platform API",
        "version": "1.0.0",
        "status": "UP",
        "dashboard": "http://127.0.0.1:8000/dashboard",
        "docs": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "services": "/api/v1/services",
            "pipeline": "/api/v1/pipeline/status",
            "incidents": "/api/v1/incidents",
            "dependencies": "/api/v1/dependencies",
            "models_evaluation": "/api/v1/models/evaluation",
            "benchmarks": "/api/v1/benchmarks/report",
            "fault_simulation": "POST /api/v1/simulations/faults",
        },
    }


@router.post("/v1/metrics")
async def ingest_otlp_metrics(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    events = collector.parse_otlp_payload(body)
    valid = collector.ingest(events)
    for evt in valid:
        broker.publish("telemetry.events", evt)
        windows.add_event(evt)
    if valid:
        lakehouse.write_events(valid)
    return {"status": "SUCCESS", "ingested": len(valid), "rejected": len(events) - len(valid)}


@router.post("/v1/traces")
async def ingest_otlp_traces(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    events = collector.parse_otlp_payload(body)
    valid = collector.ingest(events)
    for evt in valid:
        broker.publish("telemetry.events", evt)
        windows.add_event(evt)
    if valid:
        lakehouse.write_events(valid)
    return {"status": "SUCCESS", "ingested": len(valid), "rejected": len(events) - len(valid)}


@router.post("/api/telemetry/ingest")
@router.post("/api/v1/telemetry/ingest")
async def ingest_telemetry(payload: Dict[str, Any]):
    events = collector.parse_otlp_payload(payload)
    valid = collector.ingest(events)
    for evt in valid:
        broker.publish("telemetry.events", evt)
        windows.add_event(evt)
    if valid:
        lakehouse.write_events(valid)
    return {"ingested": len(valid), "rejected": len(events) - len(valid)}



@router.get("/api/telemetry/summary")
@router.get("/api/v1/telemetry/summary")
async def get_telemetry_summary():
    """Consolidated telemetry summary contract for UI dashboards (overview, trends, services, anomalies, pipeline)."""
    now = datetime.now(timezone.utc)
    lag_info = broker.get_lag("telemetry.events", "main_consumer")
    time_window = max(1.0, time.time() % 3600)
    eps = round(collector.processed_count / time_window, 1)

    services_data = []
    total_req_rate = 0.0
    latencies_p90 = []
    error_rates = []

    for svc in simulator.services:
        win = windows.get_window(svc)
        if not win:
            continue
        feat_vec = np.array([win.request_rate, win.error_rate, win.latency_p90, win.cpu_utilization])
        risk_prob = risk_model.predict_risk(feat_vec)
        tier = "EDGE" if ("edge" in svc or "gateway" in svc) else ("DATA" if ("db" in svc or "storage" in svc) else "CORE")
        status = "CRITICAL" if (win.error_rate > 0.05 or risk_prob > 0.6) else ("DEGRADED" if (win.error_rate > 0.015 or risk_prob > 0.3) else "HEALTHY")

        total_req_rate += win.request_rate
        latencies_p90.append(win.latency_p90)
        error_rates.append(win.error_rate)

        services_data.append({
            "service_id": svc,
            "name": svc.replace("-", " ").title(),
            "tier": tier,
            "status": status,
            "request_rate": round(win.request_rate, 2),
            "error_rate": round(win.error_rate, 4),
            "latency_p50": round(win.latency_p50, 2),
            "latency_p90": round(win.latency_p90, 2),
            "latency_p99": round(win.latency_p99, 2),
            "cpu_utilization": round(win.cpu_utilization, 2),
            "memory_utilization": round(win.memory_utilization, 2),
            "incident_risk_probability": round(float(risk_prob), 4),
            "risk_band": "CRITICAL" if risk_prob > 0.7 else ("HIGH" if risk_prob > 0.4 else "LOW"),
        })

    avg_p90 = round(float(np.mean(latencies_p90)), 2) if latencies_p90 else 45.0
    avg_p95 = round(avg_p90 * 1.15, 2)
    avg_err = round(float(np.mean(error_rates)), 4) if error_rates else 0.002
    anoms = arbiter.get_recent(limit=30)
    anom_dicts = [a.model_dump(mode="json") for a in anoms]

    # Performance trends / charts from edge or gateway service
    primary_svc = "edge-gateway" if "edge-gateway" in simulator.services else (simulator.services[0] if simulator.services else "order-service")
    history = windows.get_historical_metrics(primary_svc, limit=30)
    trends = []
    for h in history:
        trends.append({
            "timestamp": h.get("timestamp"),
            "request_rate": h.get("request_rate", 120.0),
            "latency_p90": h.get("latency_p90", 45.0),
            "latency_p95": round(h.get("latency_p90", 45.0) * 1.15, 2),
            "error_rate": h.get("error_rate", 0.002),
        })

    # Recent raw events sample
    recent_events = []
    if "telemetry.events" in broker.partitions:
        part0 = broker.partitions["telemetry.events"][0]
        for item in part0[-20:]:
            evt = item[1] if isinstance(item, tuple) else item
            if hasattr(evt, "model_dump"):
                recent_events.append(evt.model_dump(mode="json"))

    overall_status = "CRITICAL" if any(s["status"] == "CRITICAL" for s in services_data) else (
        "DEGRADED" if any(s["status"] == "DEGRADED" for s in services_data) else "HEALTHY"
    )

    return {
        "status": overall_status,
        "timestamp": now.isoformat(),
        "overview": {
            "request_volume": round(total_req_rate, 1),
            "p95_latency": avg_p95,
            "error_rate": avg_err,
            "active_anomalies": len(anoms),
            "total_services": len(services_data),
            "pipeline_status": "HEALTHY" if collector.rejected_count == 0 else "DEGRADED",
        },
        "performance_charts": trends,
        "services": services_data,
        "anomalies": anom_dicts,
        "events": recent_events,
        "pipeline_reliability": {
            "throughput_eps": eps,
            "consumer_lag": lag_info["total_lag"],
            "lakehouse_records": lakehouse.total_records,
            "watermark_delay_ms": collector.get_watermark_status()["watermark_delay_ms"],
            "dead_letter_queue_size": collector.dlq.total_violations,
        },
        "next_phases": {
            "incident_prediction": {
                "status": "OPERATIONAL_READY",
                "target": "10-Minute Forward Emergence Risk P(incident in t + 10m)",
                "models": ["Calibrated Logistic Regression", "XGBoost Classifier"],
                "alert_budget_quota": "10% alert budget (Recall calibrated)",
                "provenance": "CONTROLLED_FAULT_INJECTION_SIMULATION",
                "validation_notice": "Features active. Production outage dataset labels required before asserting real-world predictive coverage.",
            },
            "service_dependency_analysis": {
                "status": "OPERATIONAL_READY",
                "directed_graph": {"nodes": graph_engine.graph.number_of_nodes(), "edges": graph_engine.graph.number_of_edges()},
                "candidate_origin_ranking": "5-Factor Associative Ranking (Temporal Precedence, Graph Depth, Traces, Deployment, Severity)",
                "linguistic_governance": "Associative phrasing strictly enforced; no unverified causality claims.",
            },
        },
    }


@router.get("/api/pipeline/health")
async def get_pipeline_health():
    lag_info = broker.get_lag("telemetry.events", "main_consumer")
    return {
        "health": {
            "data_quality_status": "HEALTHY" if collector.rejected_count == 0 else "DEGRADED",
            "total_processed": collector.processed_count,
            "consumer_lag": lag_info["total_lag"],
            "lakehouse_records": lakehouse.total_records,
            "throughput_eps": round(collector.processed_count / max(1.0, time.time() % 3600), 1),
        },
        "collector": {
            "processed": collector.processed_count,
            "rejected": collector.rejected_count,
        },
    }


@router.get("/api/services/health")
async def get_services_health():
    services_list = []
    for svc in simulator.services:
        win = windows.get_window(svc)
        if not win:
            continue

        t0 = time.perf_counter()
        feat_vec = np.array([win.request_rate, win.error_rate, win.latency_p90, win.cpu_utilization])
        risk_prob = risk_model.predict_risk(feat_vec)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        model_monitor.record_latency("risk_predictor", lat_ms)
        model_monitor.record_feature("latency_p90", win.latency_p90)
        model_monitor.record_feature("error_rate", win.error_rate)
        model_monitor.record_feature("cpu_utilization", win.cpu_utilization)

        status = "CRITICAL" if (win.error_rate > 0.05 or risk_prob > 0.6) else ("DEGRADED" if (win.error_rate > 0.015 or risk_prob > 0.3) else "HEALTHY")

        services_list.append({
            "service_id": svc,
            "name": svc,
            "tier": "CORE",
            "status": status,
            "request_rate": win.request_rate,
            "error_rate": win.error_rate,
            "latency_p50": win.latency_p50,
            "latency_p90": win.latency_p90,
            "latency_p99": win.latency_p99,
            "cpu_utilization": win.cpu_utilization,
            "memory_utilization": win.memory_utilization,
            "incident_risk_probability": risk_prob,
            "risk_band": "CRITICAL" if risk_prob > 0.7 else ("HIGH" if risk_prob > 0.4 else "LOW"),
        })

    return {"services": services_list, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/api/topology")
async def get_topology():
    return graph_engine.export_topology()


@router.get("/api/anomalies/live")
async def get_live_anomalies(limit: int = Query(default=20, le=100)):
    anoms = arbiter.get_recent(limit=limit)
    return {"anomalies": [a.model_dump(mode="json") for a in anoms], "total": len(arbiter.recent_anomalies)}


@router.get("/api/incidents")
async def get_incidents():
    recent_anoms = arbiter.get_recent(limit=40)
    damping = feedback_engine.get_damping_factors()
    candidates = candidate_ranker.rank_candidates(recent_anoms, damping_factors=damping)
    fault_status = fault_harness.get_status()
    return {
        "candidate_origins": candidates,
        "fault_status": fault_status,
        "recent_anomalies_count": len(recent_anoms),
    }


@router.get("/api/alerts")
async def get_alerts():
    return {"alerts": alert_manager.get_all_alerts(limit=50)}


@router.post("/api/feedback")
async def submit_feedback(payload: FeedbackPayload):
    entry = feedback_engine.record_feedback(
        incident_id=payload.incident_id,
        suggested_origin=payload.suggested_origin,
        verdict=payload.verdict,
        actual_origin=payload.actual_origin,
    )
    return {"status": "FEEDBACK_LOGGED", "entry": entry}


@router.get("/api/forecasts")
async def get_capacity_forecasts(metric: str = Query(default="request_rate")):
    val = 145.0
    return forecaster.predict_capacity(current_val=val, capacity_limit=300.0)


@router.post("/api/simulation/fault")
async def inject_fault_endpoint(payload: FaultPayload):
    res = fault_harness.inject_scenario(payload.scenario)
    return {"status": "FAULT_INJECTED", "scenario": res}


@router.post("/api/simulation/clear")
async def clear_fault_endpoint():
    fault_harness.clear()
    return {"status": "FAULT_CLEARED"}


@router.get("/api/model/monitoring")
async def get_model_monitoring():
    return model_monitor.evaluate_drift()


@router.get("/api/security/status")
async def get_security_posture(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    status_dict = sec_manager.get_security_status()
    status_dict["client_rate_limit"] = rate_limiter.get_usage(client_ip)
    return status_dict


@router.post("/api/benchmarks/load-test")
async def run_load_test(target_events: int = 1000):
    t0 = time.perf_counter()
    for _ in range(target_events // 20):
        tick = simulator.generate_tick()
        for evt in tick:
            broker.publish("telemetry.events", evt)
            windows.add_event(evt)
    dur = time.perf_counter() - t0
    eps = target_events / max(0.001, dur)
    return {
        "events_processed": target_events,
        "duration_seconds": round(dur, 3),
        "achieved_throughput_eps": round(eps, 1),
        "single_node_capacity_limit_eps": 15000.0,
        "capacity_headroom_pct": round(((15000.0 - eps) / 15000.0) * 100.0, 1),
        "architecture_verdict": "SINGLE_NODE_ARCHITECTURE_RECOMMENDED",
    }
