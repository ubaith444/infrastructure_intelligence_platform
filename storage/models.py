from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EnvironmentType(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    SYNTHETIC_SCALE = "synthetic-scale-test"


class EventType(str, Enum):
    SERVICE_METRIC = "service_metric"
    SYSTEM_METRIC = "system_metric"
    LOG_EVENT = "log_event"
    TRACE_SPAN = "trace_span"
    STATUS_FEED = "status_feed"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    PENDING = "PENDING"
    FIRING = "FIRING"
    SUPPRESSED = "SUPPRESSED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class IncidentStatus(str, Enum):
    DETECTED = "DETECTED"
    INVESTIGATING = "INVESTIGATING"
    MITIGATED = "MITIGATED"
    RESOLVED = "RESOLVED"


class AlertRecord(BaseModel):
    alert_id: str
    service_id: str
    anomaly_id: str
    status: AlertStatus
    severity: AlertSeverity = AlertSeverity.WARNING
    suppression_reason: Optional[str] = None
    timestamp: datetime
    score: float = 0.0


class TelemetryEvent(BaseModel):
    event_id: str
    event_time: datetime
    ingested_at: Optional[datetime] = None
    service: str
    environment: str = "production"
    event_type: str = "service_metric"
    metric_name: str
    metric_value: float
    unit: Optional[str] = None
    host_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    schema_version: str = "1.0"
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def validate_timestamp_skew(cls, v: datetime) -> datetime:
        now = datetime.now(timezone.utc)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v > now and (v - now).total_seconds() > 600.0:
            raise ValueError(f"Clock skew detected: event_time {v} is >10m in future")
        return v

    @field_validator("attributes")
    @classmethod
    def validate_label_cardinality(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if len(v) > 25:
            raise ValueError("Attribute label cardinality exceeded limit (>25 tags)")
        banned = {"user_id", "email", "raw_url", "token", "password"}
        for k in v:
            if k.lower() in banned:
                raise ValueError(f"Banned unbounded attribute label detected: {k}")
        return v



class MetricWindow(BaseModel):
    service_id: str
    window_start: datetime
    window_end: datetime
    sample_count: int = 100
    request_rate: float
    error_rate: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    cpu_utilization: float
    memory_utilization: float


class AnomalyRecord(BaseModel):
    anomaly_id: str
    service_id: str
    detector_type: str
    timestamp: datetime
    severity: AlertSeverity
    score: float
    threshold: float = 3.5
    metric_attributions: Dict[str, float] = Field(default_factory=dict)
    narrative: str = ""


class IncidentRecord(BaseModel):
    incident_id: str
    title: str
    status: IncidentStatus = IncidentStatus.DETECTED
    started_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    service_count: int = 1
    services_involved: List[str] = Field(default_factory=list)
    primary_candidate_origin: Optional[str] = None
    origin_confidence: float = 0.0
    timeline_events: List[Dict[str, Any]] = Field(default_factory=list)


class ReplayStatusEnum(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"


class LateEventPolicy(str, Enum):
    DROP = "DROP"
    ROUTE_TO_DLQ = "ROUTE_TO_DLQ"
    PROCESS_LATE = "PROCESS_LATE"


class ReplayRequest(BaseModel):
    dataset_or_source: str = "lakehouse"
    service_id: Optional[str] = None
    metric_name: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    speed_factor: float = Field(default=1.0, gt=0.0, le=100.0)
    target_topic: str = "telemetry.events"


class ReplayStatus(BaseModel):
    session_id: str
    status: ReplayStatusEnum = ReplayStatusEnum.IDLE
    dataset_or_source: str
    total_events: int = 0
    events_replayed: int = 0
    progress_pct: float = 0.0
    speed_factor: float = 1.0
    started_at: Optional[datetime] = None
    current_event_time: Optional[datetime] = None
    target_topic: str = "telemetry.events"


# ============================================================================
# PHASE 4 CORE RELATIONAL ENTITIES (PostgreSQL / Metadata Store)
# ============================================================================

class ServiceEntity(BaseModel):
    service_id: str
    name: str
    environment: str = "production"
    owner: str = "platform-team"
    tier: str = "CORE"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ServiceDependencyEntity(BaseModel):
    source_service: str
    target_service: str
    relationship: str = "synchronous_http"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryEventIndex(BaseModel):
    event_id: str
    event_time: datetime
    service: str
    event_type: str
    payload_reference: str
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetricWindowEntity(BaseModel):
    service_id: str
    window_start: datetime
    request_rate: float
    error_rate: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0


class AnomalyEventEntity(BaseModel):
    model_config = {"protected_namespaces": ()}
    anomaly_id: str
    service_id: str
    score: float
    threshold: float
    model_version: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IncidentEntity(BaseModel):
    incident_id: str
    title: str
    severity: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    service_count: int = 1
    primary_candidate_origin: Optional[str] = None
    origin_confidence: float = 0.0


class IncidentEvidenceEntity(BaseModel):
    incident_id: str
    related_anomalies: List[str] = Field(default_factory=list)
    contributing_signals: List[Dict[str, Any]] = Field(default_factory=list)
    associative_evidence: str = ""


class ModelRunEntity(BaseModel):
    model_config = {"protected_namespaces": ()}
    run_id: str
    version: str
    model_name: str
    training_period: str
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict)
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))



class PipelineHealthEntity(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    throughput: float
    lag: int
    errors: int
    data_quality_status: str


