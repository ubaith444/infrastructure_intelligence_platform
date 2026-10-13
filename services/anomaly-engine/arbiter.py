from collections import deque
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
from services.anomaly_engine.isolation_forest import MultivariateIsolationForest
from services.anomaly_engine.statistical_mad import RobustStatisticalDetector
from services.anomaly_engine.streaming_halfspace import RiverStreamingDetector
from storage.models import AlertSeverity, AnomalyRecord, MetricWindow


class MultiModelAnomalyArbiter:
    """Evaluates telemetry windows concurrently across statistical, multivariate ML, and streaming detectors."""

    def __init__(self):
        self.mad = RobustStatisticalDetector()
        self.iforest = MultivariateIsolationForest()
        self.streaming = RiverStreamingDetector()
        self.recent_anomalies: deque = deque(maxlen=200)

    def evaluate_window(self, window: MetricWindow, active_incident: bool = False) -> List[AnomalyRecord]:
        detected = []
        ts = window.window_end or datetime.now(timezone.utc)

        # 1. Statistical MAD on latency and errors with seasonality
        z_lat, med_lat, anom_lat = self.mad.score(window.service_id, "latency", window.latency_p90, timestamp=ts)
        z_err, med_err, anom_err = self.mad.score(window.service_id, "error_rate", window.error_rate, timestamp=ts)
        z_cpu, med_cpu, anom_cpu = self.mad.score(window.service_id, "cpu", window.cpu_utilization, timestamp=ts)

        if anom_lat or anom_err or anom_cpu:
            sev = AlertSeverity.CRITICAL if (z_err > 5.0 or z_lat > 6.0) else AlertSeverity.WARNING
            total_z = z_lat + z_err + z_cpu + 1e-6
            mad_attributions = {
                "latency_p90": round(z_lat / total_z, 3),
                "error_rate": round(z_err / total_z, 3),
                "cpu_utilization": round(z_cpu / total_z, 3),
            }
            anom = AnomalyRecord(
                anomaly_id=f"anom-mad-{uuid.uuid4().hex[:8]}",
                service_id=window.service_id,
                detector_type="RobustStatisticalMAD",
                timestamp=ts,
                severity=sev,
                score=max(z_lat, z_err, z_cpu),
                threshold=3.5,
                metric_attributions=mad_attributions,
                narrative=f"Statistical deviation detected on {window.service_id}: latency_z={z_lat}, err_z={z_err}, cpu_z={z_cpu}",
            )
            detected.append(anom)
            self.recent_anomalies.append(anom)

        # 2. Multivariate Isolation Forest with subspace attribution
        if_score, if_anom, if_attrib = self.iforest.score_window(window)
        if if_anom:
            anom = AnomalyRecord(
                anomaly_id=f"anom-if-{uuid.uuid4().hex[:8]}",
                service_id=window.service_id,
                detector_type="MultivariateIsolationForest",
                timestamp=ts,
                severity=AlertSeverity.WARNING,
                score=if_score,
                threshold=0.65,
                metric_attributions=if_attrib,
                narrative=f"Multivariate subspace anomaly detected on {window.service_id} (score {if_score:.2f})",
            )
            detected.append(anom)
            self.recent_anomalies.append(anom)

        # 3. River Streaming with ADWIN drift detector and incident freeze
        st_score, st_anom, drift_detected, st_attrib = self.streaming.score_and_learn(
            window, is_incident_active=active_incident
        )
        if st_anom or drift_detected:
            anom = AnomalyRecord(
                anomaly_id=f"anom-river-{uuid.uuid4().hex[:8]}",
                service_id=window.service_id,
                detector_type="RiverHalfSpaceTrees",
                timestamp=ts,
                severity=AlertSeverity.WARNING,
                score=st_score,
                threshold=0.70,
                metric_attributions=st_attrib,
                narrative=f"Streaming distribution shift detected on {window.service_id} (score {st_score:.2f}, drift={drift_detected})",
            )
            detected.append(anom)
            self.recent_anomalies.append(anom)

        return detected

    def compare_on_batch(self, windows: List[MetricWindow], active_incident: bool = False) -> Dict[str, Any]:
        """Compares all 3 detectors concurrently over an identical batch of observation windows."""
        mad_detections = 0
        if_detections = 0
        river_detections = 0
        total = len(windows)

        for win in windows:
            # MAD
            _, _, a_lat = self.mad.score(win.service_id, "latency", win.latency_p90, timestamp=win.window_end)
            _, _, a_err = self.mad.score(win.service_id, "error_rate", win.error_rate, timestamp=win.window_end)
            if a_lat or a_err:
                mad_detections += 1

            # IF
            _, a_if, _ = self.iforest.score_window(win)
            if a_if:
                if_detections += 1

            # River
            _, a_st, drift, _ = self.streaming.score_and_learn(win, is_incident_active=active_incident)
            if a_st or drift:
                river_detections += 1

        return {
            "total_windows_evaluated": total,
            "statistical_mad_anomaly_rate": round(mad_detections / max(1, total), 4),
            "isolation_forest_anomaly_rate": round(if_detections / max(1, total), 4),
            "river_streaming_anomaly_rate": round(river_detections / max(1, total), 4),
            "safeguards": {
                "mad_zero_fallback_active": True,
                "river_incident_freeze_active": active_incident,
                "frozen_update_count": self.streaming.freeze_count,
            },
        }

    def get_recent(self, limit: int = 50) -> List[AnomalyRecord]:
        return list(self.recent_anomalies)[-limit:]

