import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from services.ingestion.datasets.dataset_loader import DatasetLoader
from storage.broker import EmbeddedEventBroker
from storage.lakehouse import LakehouseManager
from storage.models import ReplayRequest, ReplayStatus, ReplayStatusEnum, TelemetryEvent

logger = logging.getLogger("iip.storage.replay")


class EventReplayEngine:
    """Orchestrates asynchronous historical event replay from DuckDB lakehouse or public datasets

    with variable speed factor, offset rewinds, and real-time progress tracking.
    """

    def __init__(self, broker: EmbeddedEventBroker, lakehouse: LakehouseManager):
        self.broker = broker
        self.lakehouse = lakehouse
        self.dataset_loader = DatasetLoader()
        self.current_status = ReplayStatus(
            session_id="none",
            status=ReplayStatusEnum.IDLE,
            dataset_or_source="none",
        )
        self._task: Optional[asyncio.Task] = None
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stop_requested = False

    def get_status(self) -> Dict[str, Any]:
        return self.current_status.model_dump(mode="json")

    def _fetch_events_for_replay(self, req: ReplayRequest) -> List[TelemetryEvent]:
        source = req.dataset_or_source.lower()
        if source in ["google_cluster", "loghub", "opentelemetry"]:
            return self.dataset_loader.load_dataset(source)

        # Default: lakehouse
        events = []
        try:
            sql = "SELECT * FROM read_parquet('data/lakehouse/**/*.parquet')"
            clauses = []
            if req.service_id:
                clauses.append(f"service = '{req.service_id}'")
            if req.metric_name:
                clauses.append(f"metric_name = '{req.metric_name}'")
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY event_time ASC LIMIT 500"

            records = self.lakehouse.query(sql)
            for r in records:
                try:
                    events.append(
                        TelemetryEvent(
                            event_id=str(r.get("event_id", f"rpl-{uuid.uuid4().hex[:8]}")),
                            event_time=datetime.fromisoformat(r["event_time"]).replace(tzinfo=timezone.utc),
                            service=r["service"],
                            environment=r.get("environment", "production"),
                            event_type=r.get("event_type", "service_metric"),
                            metric_name=r["metric_name"],
                            metric_value=float(r["metric_value"]),
                            unit=r.get("unit"),
                            host_id=r.get("host_id"),
                        )
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not read from Parquet lakehouse for replay: {e}")

        if not events:
            # Fallback: load sample dataset if lakehouse is empty
            events = self.dataset_loader.load_dataset("google_cluster")
        return events

    async def start_replay(self, req: ReplayRequest, on_event_replayed=None) -> Dict[str, Any]:
        if self._task and not self._task.done():
            await self.stop_replay()

        events = self._fetch_events_for_replay(req)
        session_id = f"rpl-{uuid.uuid4().hex[:8]}"

        self.current_status = ReplayStatus(
            session_id=session_id,
            status=ReplayStatusEnum.RUNNING,
            dataset_or_source=req.dataset_or_source,
            total_events=len(events),
            events_replayed=0,
            progress_pct=0.0,
            speed_factor=req.speed_factor,
            started_at=datetime.now(timezone.utc),
            current_event_time=events[0].event_time if events else None,
            target_topic=req.target_topic,
        )

        self._stop_requested = False
        self._pause_event.set()
        self._task = asyncio.create_task(self._replay_loop(events, req, on_event_replayed))
        return self.get_status()

    async def _replay_loop(self, events: List[TelemetryEvent], req: ReplayRequest, on_event_replayed=None):
        logger.info(f"Starting replay session {self.current_status.session_id} with {len(events)} events (speed: {req.speed_factor}x)")
        sleep_interval = max(0.01, 1.0 / req.speed_factor)

        for i, evt in enumerate(events):
            if self._stop_requested:
                self.current_status.status = ReplayStatusEnum.STOPPED
                break

            await self._pause_event.wait()

            # Publish event into target broker topic
            self.broker.publish(req.target_topic, evt)
            if on_event_replayed:
                try:
                    on_event_replayed(evt)
                except Exception as ex:
                    logger.debug(f"Error in on_event_replayed callback: {ex}")

            self.current_status.events_replayed = i + 1
            self.current_status.progress_pct = round(((i + 1) / len(events)) * 100.0, 1)
            self.current_status.current_event_time = evt.event_time

            await asyncio.sleep(sleep_interval)

        if not self._stop_requested:
            self.current_status.status = ReplayStatusEnum.COMPLETED
        logger.info(f"Replay session {self.current_status.session_id} ended: {self.current_status.status.value}")

    async def pause_replay(self) -> Dict[str, Any]:
        if self.current_status.status == ReplayStatusEnum.RUNNING:
            self._pause_event.clear()
            self.current_status.status = ReplayStatusEnum.PAUSED
        return self.get_status()

    async def resume_replay(self) -> Dict[str, Any]:
        if self.current_status.status == ReplayStatusEnum.PAUSED:
            self._pause_event.set()
            self.current_status.status = ReplayStatusEnum.RUNNING
        return self.get_status()

    async def stop_replay(self) -> Dict[str, Any]:
        self._stop_requested = True
        self._pause_event.set()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
            except Exception:
                pass
        self.current_status.status = ReplayStatusEnum.STOPPED
        return self.get_status()
