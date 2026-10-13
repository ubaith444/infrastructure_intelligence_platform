from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("iip.storage.metadata_store")


class OperationalMetadataStore:
    """Relational metadata store managing operational inventory, dependencies, alerts, and incident evidence.
    
    Architecture:
        - Operational metadata (services, dependencies, incidents, alerts, model runs) stored relationally.
        - High-volume raw telemetry streams retained in partitioned DuckDB Parquet lakehouse.
        - Provides SQLite engine by default with zero setup, with PostgreSQL connectivity via DATABASE_URL.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/operational_metadata.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dsn = os.getenv("DATABASE_URL", str(self.db_path))
        self._init_schema()
        self._seed_default_services_and_dependencies()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initializes tables for all 9 core operational entities."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Services
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS services (
                    service_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    environment TEXT NOT NULL DEFAULT 'production',
                    owner TEXT NOT NULL DEFAULT 'platform-team',
                    tier TEXT NOT NULL DEFAULT 'CORE',
                    created_at TEXT NOT NULL
                )
            """)

            # 2. Service Dependencies
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_service TEXT NOT NULL,
                    target_service TEXT NOT NULL,
                    relationship TEXT NOT NULL DEFAULT 'synchronous_http',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_service, target_service)
                )
            """)

            # 3. Telemetry Event Index (Reference to lakehouse payload)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_event_index (
                    event_id TEXT PRIMARY KEY,
                    event_time TEXT NOT NULL,
                    service TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_reference TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                )
            """)

            # 4. Metric Windows
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metric_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    request_rate REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    latency_p50 REAL NOT NULL,
                    latency_p90 REAL NOT NULL,
                    latency_p99 REAL NOT NULL,
                    cpu_utilization REAL NOT NULL DEFAULT 0.0,
                    memory_utilization REAL NOT NULL DEFAULT 0.0
                )
            """)

            # 5. Anomaly Events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    anomaly_id TEXT PRIMARY KEY,
                    service_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            # 6. Incidents
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    service_count INTEGER NOT NULL DEFAULT 1,
                    primary_candidate_origin TEXT,
                    origin_confidence REAL NOT NULL DEFAULT 0.0,
                    candidate_rankings_json TEXT,
                    timeline_json TEXT
                )
            """)

            # 7. Incident Evidence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incident_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL,
                    related_anomalies_json TEXT NOT NULL,
                    contributing_signals_json TEXT NOT NULL,
                    associative_evidence TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                )
            """)

            # 8. Model Runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    training_period TEXT NOT NULL,
                    evaluation_metrics_json TEXT NOT NULL,
                    trained_at TEXT NOT NULL
                )
            """)

            # 9. Pipeline Health
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    throughput REAL NOT NULL,
                    lag INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    data_quality_status TEXT NOT NULL
                )
            """)

            # 10. Alerts Management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    service_id TEXT NOT NULL,
                    anomaly_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'WARNING',
                    suppression_reason TEXT,
                    timestamp TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    acknowledged_by TEXT
                )
            """)

            # 11. Analyst Feedback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyst_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    suggested_origin TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    actual_origin TEXT,
                    recorded_at TEXT NOT NULL
                )
            """)

            conn.commit()

    def _seed_default_services_and_dependencies(self) -> None:
        """Seeds initial default services and dependency graph topology."""
        default_services = [
            ("edge-gateway", "Edge Gateway", "production", "networking-team", "EDGE"),
            ("checkout-api", "Checkout API", "production", "checkout-team", "CORE"),
            ("order-service", "Order Service", "production", "orders-team", "CORE"),
            ("payment-service", "Payment Service", "production", "payments-team", "CORE"),
            ("inventory-service", "Inventory Service", "production", "inventory-team", "CORE"),
            ("auth-service", "Auth Service", "production", "security-team", "CORE"),
            ("db-primary", "Primary PostgreSQL DB", "production", "data-team", "DATA"),
            ("redis-cache", "Redis Cache", "production", "data-team", "DATA"),
        ]
        default_deps = [
            ("edge-gateway", "checkout-api", "synchronous_http"),
            ("checkout-api", "order-service", "synchronous_http"),
            ("checkout-api", "auth-service", "synchronous_http"),
            ("checkout-api", "payment-service", "synchronous_http"),
            ("order-service", "inventory-service", "synchronous_http"),
            ("order-service", "db-primary", "tcp_pool"),
            ("payment-service", "db-primary", "tcp_pool"),
            ("auth-service", "redis-cache", "tcp_cache"),
            ("inventory-service", "db-primary", "tcp_pool"),
        ]

        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for sid, name, env, owner, tier in default_services:
                cursor.execute("""
                    INSERT OR IGNORE INTO services (service_id, name, environment, owner, tier, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (sid, name, env, owner, tier, now_str))

            for src, tgt, rel in default_deps:
                cursor.execute("""
                    INSERT OR IGNORE INTO service_dependencies (source_service, target_service, relationship, created_at)
                    VALUES (?, ?, ?, ?)
                """, (src, tgt, rel, now_str))
            conn.commit()

    # --- SERVICES & DEPENDENCIES ---

    def list_services(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM services ORDER BY service_id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def list_dependencies(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM service_dependencies ORDER BY source_service ASC")
            return [dict(r) for r in cursor.fetchall()]

    # --- INCIDENTS & EVIDENCE ---

    def save_incident(self, incident_dict: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO incidents (
                    incident_id, title, severity, status, start_time, end_time,
                    service_count, primary_candidate_origin, origin_confidence,
                    candidate_rankings_json, timeline_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_dict["incident_id"],
                incident_dict["title"],
                incident_dict.get("severity", "WARNING"),
                incident_dict.get("status", "DETECTED"),
                incident_dict.get("started_at") or incident_dict.get("start_time") or datetime.now(timezone.utc).isoformat(),
                incident_dict.get("resolved_at") or incident_dict.get("end_time"),
                incident_dict.get("service_count", 1),
                incident_dict.get("primary_candidate_origin"),
                float(incident_dict.get("origin_confidence", 0.0)),
                json.dumps(incident_dict.get("candidate_rankings", [])),
                json.dumps(incident_dict.get("timeline", [])),
            ))
            conn.commit()

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            res["candidate_rankings"] = json.loads(res.pop("candidate_rankings_json", "[]") or "[]")
            res["timeline"] = json.loads(res.pop("timeline_json", "[]") or "[]")
            return res

    def list_incidents(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        if severity:
            query += " AND severity = ?"
            params.append(severity.upper())
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["candidate_rankings"] = json.loads(item.pop("candidate_rankings_json", "[]") or "[]")
                item["timeline"] = json.loads(item.pop("timeline_json", "[]") or "[]")
                results.append(item)
            return results

    def save_incident_evidence(
        self,
        incident_id: str,
        related_anomalies: List[str],
        contributing_signals: List[Dict[str, Any]],
        associative_evidence: str,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incident_evidence (
                    incident_id, related_anomalies_json, contributing_signals_json,
                    associative_evidence, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                incident_id,
                json.dumps(related_anomalies),
                json.dumps(contributing_signals),
                associative_evidence,
                datetime.now(timezone.utc).isoformat(),
            ))
            conn.commit()

    def get_incident_evidence(self, incident_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incident_evidence WHERE incident_id = ? ORDER BY id DESC", (incident_id,))
            rows = cursor.fetchall()
            evidences = []
            for r in rows:
                ev = dict(r)
                ev["related_anomalies"] = json.loads(ev.pop("related_anomalies_json", "[]"))
                ev["contributing_signals"] = json.loads(ev.pop("contributing_signals_json", "[]"))
                evidences.append(ev)
            return evidences

    # --- ALERTS ---

    def save_alert(self, alert_dict: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, service_id, anomaly_id, status, severity,
                    suppression_reason, timestamp, score, acknowledged_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_dict["alert_id"],
                alert_dict["service_id"],
                alert_dict["anomaly_id"],
                alert_dict["status"],
                alert_dict.get("severity", "WARNING"),
                alert_dict.get("suppression_reason"),
                alert_dict["timestamp"],
                float(alert_dict.get("score", 0.0)),
                alert_dict.get("acknowledged_by"),
            ))
            conn.commit()

    def list_alerts(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def acknowledge_alert(self, alert_id: str, user: str = "oncall-analyst") -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts SET status = 'ACKNOWLEDGED', acknowledged_by = ? WHERE alert_id = ?
            """, (user, alert_id))
            conn.commit()
            return cursor.rowcount > 0

    # --- ANALYST FEEDBACK ---

    def save_analyst_feedback(
        self,
        incident_id: str,
        suggested_origin: str,
        verdict: str,
        actual_origin: Optional[str] = None,
    ) -> Dict[str, Any]:
        fid = f"fb-{uuid.uuid4().hex[:8]}"
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyst_feedback (
                    feedback_id, incident_id, suggested_origin, verdict, actual_origin, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (fid, incident_id, suggested_origin, verdict, actual_origin or suggested_origin, now_str))
            conn.commit()
        return {
            "feedback_id": fid,
            "incident_id": incident_id,
            "suggested_origin": suggested_origin,
            "verdict": verdict,
            "actual_origin": actual_origin or suggested_origin,
            "recorded_at": now_str,
        }

    # --- MODEL RUNS ---

    def record_model_run(
        self,
        model_name: str,
        version: str,
        training_period: str,
        evaluation_metrics: Dict[str, Any],
    ) -> str:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_runs (
                    run_id, version, model_name, training_period, evaluation_metrics_json, trained_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, version, model_name, training_period, json.dumps(evaluation_metrics), now_str))
            conn.commit()
        return run_id

    def list_model_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_runs ORDER BY trained_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            runs = []
            for r in rows:
                item = dict(r)
                item["evaluation_metrics"] = json.loads(item.pop("evaluation_metrics_json", "{}"))
                runs.append(item)
            return runs

    # --- PIPELINE HEALTH ---

    def record_pipeline_health(
        self,
        throughput: float,
        lag: int,
        errors: int,
        status: str,
    ) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pipeline_health (timestamp, throughput, lag, errors, data_quality_status)
                VALUES (?, ?, ?, ?, ?)
            """, (now_str, throughput, lag, errors, status))
            conn.commit()
