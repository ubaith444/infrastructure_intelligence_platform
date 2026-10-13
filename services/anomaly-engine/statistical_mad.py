from collections import deque
from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from storage.models import AlertSeverity, AnomalyRecord, MetricWindow


class RobustStatisticalDetector:
    """Robust rolling Median and Median Absolute Deviation (MAD) detector with seasonality-aware thresholds.
    
    Formula:
        MAD = median(|x_i - m|)
        z_r = 0.6745 * (x - m) / MAD
    """

    def __init__(self, history_size: int = 100, threshold_z: float = 3.5):
        self.history_size = history_size
        self.threshold_z = threshold_z
        # service -> metric_name -> deque of (timestamp, value)
        self.history: Dict[str, Dict[str, deque]] = {}
        # Diurnal seasonality factors (24 hours) - baseline multiplier based on typical business hours
        self.hourly_seasonality: Dict[int, float] = {
            h: 0.6 + 0.8 * math.sin(math.pi * (h - 6) / 12.0) if 6 <= h <= 18 else 0.5
            for h in range(24)
        }

    def score(
        self,
        service_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[float, float, bool]:
        """Compute robust z-score using MAD with zero-division safeguard and seasonal adjustment.
        
        Returns:
            Tuple of (robust_z_score, median, is_anomaly)
        """
        ts = timestamp or datetime.now(timezone.utc)
        if service_id not in self.history:
            self.history[service_id] = {}
        if metric_name not in self.history[service_id]:
            self.history[service_id][metric_name] = deque(maxlen=self.history_size)

        buf = self.history[service_id][metric_name]
        buf.append((ts, value))

        if len(buf) < 10:
            return 0.0, float(value), False

        values = np.array([pt[1] for pt in buf], dtype=float)
        med = float(np.median(values))
        abs_diffs = np.abs(values - med)
        mad = float(np.median(abs_diffs))

        # Zero-MAD safety fallback: prevent division by zero on static or low-entropy series
        if mad <= 1e-6:
            std = float(np.std(values))
            scale = std if std > 1e-6 else max(1e-4, 0.01 * max(1e-4, abs(med)))
            # Compute equivalent scale where scale = mad / 0.6745
            robust_z = 0.6745 * abs(value - med) / (scale / 1.4826)
        else:
            robust_z = 0.6745 * abs(value - med) / mad

        # Seasonality-aware threshold adjustment
        hour = ts.hour if hasattr(ts, "hour") else 12
        seasonal_factor = self.hourly_seasonality.get(hour, 1.0)
        # In low-traffic hours (seasonal_factor < 0.7), sensitivity is adjusted to prevent false alarms
        effective_threshold = self.threshold_z * (1.1 if seasonal_factor < 0.7 else 1.0)

        is_anom = robust_z >= effective_threshold
        return float(round(robust_z, 3)), float(round(med, 3)), is_anom

