from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("iip.services.incident_engine.feedback")


class AnalystFeedbackEngine:
    """Collects operator feedback on origin attributions and calibrates empirical damping factors."""

    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = feedback_dir or Path("data/feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "analyst_feedback.jsonl"
        self.confirmed_origins = 0
        self.false_positives = 0
        self.incorrect_origins = 0
        self.service_feedback_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_feedback(self, incident_id: str, suggested_origin: str, verdict: str, actual_origin: Optional[str] = None) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "incident_id": incident_id,
            "suggested_origin": suggested_origin,
            "verdict": verdict,
            "actual_origin": actual_origin or suggested_origin,
        }
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if verdict == "CONFIRMED":
            self.confirmed_origins += 1
            self.service_feedback_counts[suggested_origin]["confirmed"] += 1
        elif verdict == "FALSE_POSITIVE":
            self.false_positives += 1
            self.service_feedback_counts[suggested_origin]["false_positive"] += 1
        elif verdict == "INCORRECT_ORIGIN":
            self.incorrect_origins += 1
            self.service_feedback_counts[suggested_origin]["incorrect"] += 1

        return entry

    def get_damping_factors(self) -> Dict[str, float]:
        damping = {}
        for svc, counts in self.service_feedback_counts.items():
            conf = counts.get("confirmed", 0)
            fp = counts.get("false_positive", 0)
            if (conf + fp) > 0:
                acc = conf / (conf + fp)
                damping[svc] = max(0.2, min(1.0, 0.5 + 0.5 * acc))
        return damping
