from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from storage.models import TelemetryEvent

logger = logging.getLogger("iip.storage.lakehouse")


class LakehouseManager:
    """Analytical columnar Parquet lakehouse powered by DuckDB SQL."""

    def __init__(self, lakehouse_dir: Optional[Path] = None):
        self.lakehouse_dir = lakehouse_dir or Path("data/lakehouse")
        self.lakehouse_dir.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(database=":memory:")
        self.total_records = 0

    def write_events(self, events: List[TelemetryEvent]) -> int:
        if not events:
            return 0
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        partition_dir = self.lakehouse_dir / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "event_id": [e.event_id for e in events],
            "event_time": [e.event_time.isoformat() for e in events],
            "service": [e.service for e in events],
            "environment": [e.environment for e in events],
            "event_type": [e.event_type for e in events],
            "metric_name": [e.metric_name for e in events],
            "metric_value": [e.metric_value for e in events],
            "unit": [e.unit or "" for e in events],
            "host_id": [e.host_id or "" for e in events],
        }
        table = pa.Table.from_pydict(data)
        file_path = partition_dir / f"part_{int(datetime.now(timezone.utc).timestamp()*1000)}.parquet"
        pq.write_table(table, file_path, compression="snappy")
        self.total_records += len(events)
        return len(events)

    def query(self, sql: str) -> List[Dict[str, Any]]:
        try:
            rel = self.db.query(sql)
            return rel.to_df().to_dict(orient="records")
        except Exception as e:
            logger.error(f"Lakehouse SQL error: {e}")
            return []
