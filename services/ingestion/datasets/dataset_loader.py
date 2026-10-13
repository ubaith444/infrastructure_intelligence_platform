import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from services.ingestion.datasets.normalizers import (
    GoogleClusterNormalizer,
    LogHubNormalizer,
    OpenTelemetryTraceNormalizer,
)
from storage.models import TelemetryEvent

logger = logging.getLogger("iip.services.ingestion.datasets.loader")


class DatasetLoader:
    """Discovers, parses, and normalizes public datasets into standard platform TelemetryEvents."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/sample")

    def list_available_datasets(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "google_cluster",
                "name": "Google Cluster Workload Traces",
                "recommended_for": "CPU and memory anomaly detection, task scheduling, capacity planning",
                "file": "google_cluster_sample.json",
                "metrics": ["cpu_utilization", "memory_utilization", "disk_io_time_ratio"],
                "target_service": "cluster-scheduler",
            },
            {
                "id": "loghub",
                "name": "LogHub HDFS Distributed Logs",
                "recommended_for": "Log anomaly detection, failure classification, template extraction",
                "file": "loghub_hdfs_sample.json",
                "metrics": ["log_severity_weight"],
                "target_service": "distributed-storage",
            },
            {
                "id": "opentelemetry",
                "name": "OpenTelemetry Demo Distributed Traces",
                "recommended_for": "Service dependency graphs, trace span latency, cascade analysis",
                "file": "opentelemetry_traces_sample.json",
                "metrics": ["request_latency_ms"],
                "target_service": "edge-gateway, checkout-api, order-service, payment-service",
            },
        ]

    def load_dataset(self, dataset_id: str) -> List[TelemetryEvent]:
        mapping = {
            "google_cluster": ("google_cluster_sample.json", GoogleClusterNormalizer.normalize),
            "loghub": ("loghub_hdfs_sample.json", lambda r: [LogHubNormalizer.normalize(r)]),
            "opentelemetry": ("opentelemetry_traces_sample.json", lambda r: [OpenTelemetryTraceNormalizer.normalize(r)]),
        }

        cfg = mapping.get(dataset_id.lower())
        if not cfg:
            raise ValueError(f"Unknown dataset id '{dataset_id}'. Choose from: {list(mapping.keys())}")

        filename, normalizer_func = cfg
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.warning(f"Dataset file {file_path} not found.")
            return []

        events: List[TelemetryEvent] = []
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            for r in records:
                normalized = normalizer_func(r)
                events.extend(normalized)

        return events
