from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("iip.storage.dlq")


class DeadLetterQueueManager:
    """Quarantines malformed, invalid, schema-violating, or late-arriving events for auditing."""

    def __init__(self, dlq_dir: Optional[Path] = None, max_recent: int = 500):
        self.dlq_dir = dlq_dir or Path("data/dlq")
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        self.violations_file = self.dlq_dir / "violations.jsonl"
        self.total_violations = 0
        self.recent_violations: deque = deque(maxlen=max_recent)
        self._load_persisted_violations()

    def _load_persisted_violations(self) -> None:
        if self.violations_file.exists():
            try:
                with open(self.violations_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            self.recent_violations.append(entry)
                            self.total_violations += 1
            except Exception as e:
                logger.warning(f"Could not load historical DLQ violations: {e}")

    def record_violation(self, raw_payload: Any, reason: str) -> Dict[str, Any]:
        self.total_violations += 1
        entry = {
            "violation_id": f"dlq-{self.total_violations}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "raw_payload": raw_payload,
        }
        self.recent_violations.append(entry)
        try:
            with open(self.violations_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to DLQ file: {e}")
        return entry

    def get_recent_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.recent_violations)[-limit:]

    def clear(self) -> None:
        self.recent_violations.clear()
        self.total_violations = 0
        if self.violations_file.exists():
            try:
                self.violations_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete DLQ file: {e}")
