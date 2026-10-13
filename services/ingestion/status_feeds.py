from datetime import datetime, timezone
from typing import Any, Dict, List


class StatusFeedPoller:
    """Simulates external cloud status feed polling (AWS, GCP, Azure, GitHub)."""

    def __init__(self):
        self.feed_states = {
            "us-east-1": "OPERATIONAL",
            "us-west-2": "OPERATIONAL",
            "eu-central-1": "OPERATIONAL",
        }

    def get_latest_status(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regions": self.feed_states,
            "external_degradation_detected": any(s != "OPERATIONAL" for s in self.feed_states.values()),
        }
