from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from storage.models import TelemetryEvent

logger = logging.getLogger("iip.storage.broker")


class EmbeddedEventBroker:
    """Embedded durable partitioned streaming event broker with offset checkpointing,

    seek/replay capabilities, and idempotent deduplication.
    """

    def __init__(self, storage_dir: Optional[Path] = None, num_partitions: int = 4, max_dedup_window: int = 10000):
        self.storage_dir = storage_dir or Path("data/broker")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.num_partitions = num_partitions
        self.offsets_file = self.storage_dir / "consumer_offsets.json"
        self._lock = threading.RLock()

        # topic -> partition_id -> list of (offset, TelemetryEvent)
        self.partitions: Dict[str, Dict[int, List[Tuple[int, TelemetryEvent]]]] = defaultdict(lambda: defaultdict(list))
        # consumer_group -> topic -> partition_id -> committed_offset
        self.consumer_offsets: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Idempotence: sliding deduplication window
        self.seen_event_ids: deque = deque(maxlen=max_dedup_window)
        self._seen_set: Set[str] = set()

        self._load_offsets()
        self._load_persisted_logs()

    def _get_partition(self, event: TelemetryEvent) -> int:
        key = event.service or "default"
        return abs(hash(key)) % self.num_partitions

    def _get_partition_log_path(self, topic: str, partition_id: int) -> Path:
        topic_dir = self.storage_dir / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        return topic_dir / f"partition_{partition_id}.log"

    def _load_offsets(self) -> None:
        if self.offsets_file.exists():
            try:
                with open(self.offsets_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    for cg, topics in raw.items():
                        for topic, parts in topics.items():
                            for pid, off in parts.items():
                                self.consumer_offsets[cg][topic][int(pid)] = off
            except Exception as e:
                logger.warning(f"Could not load offsets: {e}")

    def _save_offsets(self) -> None:
        try:
            raw = {
                cg: {topic: {str(pid): off for pid, off in parts.items()} for topic, parts in topics.items()}
                for cg, topics in self.consumer_offsets.items()
            }
            with open(self.offsets_file, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save offsets: {e}")

    def _load_persisted_logs(self) -> None:
        """Loads persisted partition log files into memory on startup."""
        for topic_dir in self.storage_dir.iterdir():
            if topic_dir.is_dir() and not topic_dir.name.startswith("."):
                topic = topic_dir.name
                for p_file in topic_dir.glob("partition_*.log"):
                    try:
                        pid = int(p_file.stem.split("_")[-1])
                        with open(p_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line_str = line.strip()
                                if not line_str:
                                    continue
                                try:
                                    data = json.loads(line_str)
                                    evt = TelemetryEvent(**data["event"])
                                    off = data["offset"]
                                    self.partitions[topic][pid].append((off, evt))
                                    self.seen_event_ids.append(evt.event_id)
                                    self._seen_set.add(evt.event_id)
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.warning(f"Error loading partition log {p_file}: {e}")

    def publish(self, topic: str, event: TelemetryEvent, persist: bool = True) -> Tuple[int, int, bool]:
        """Publishes an event to the partitioned topic.

        Returns (partition_id, offset, is_duplicate).
        """
        with self._lock:
            # Deduplication check
            if event.event_id in self._seen_set:
                partition_id = self._get_partition(event)
                return partition_id, -1, True

            partition_id = self._get_partition(event)
            part_list = self.partitions[topic][partition_id]
            offset = len(part_list)
            part_list.append((offset, event))

            # Add to deduplication cache
            if len(self.seen_event_ids) >= self.seen_event_ids.maxlen:
                evicted = self.seen_event_ids.popleft()
                self._seen_set.discard(evicted)
            self.seen_event_ids.append(event.event_id)
            self._seen_set.add(event.event_id)

            # Append to durable on-disk log
            if persist:
                try:
                    log_path = self._get_partition_log_path(topic, partition_id)
                    entry = {"offset": offset, "event": event.model_dump(mode="json")}
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    logger.error(f"Failed to persist partition log: {e}")

            return partition_id, offset, False

    def poll(
        self,
        topic: str,
        consumer_group: str,
        max_records: int = 100,
        auto_commit: bool = True,
    ) -> List[TelemetryEvent]:
        """Polls records for the given consumer group across assigned partitions."""
        with self._lock:
            records = []
            for pid in range(self.num_partitions):
                part_list = self.partitions[topic][pid]
                current_offset = self.consumer_offsets[consumer_group][topic][pid]
                available = part_list[current_offset: current_offset + (max_records - len(records))]
                for off, evt in available:
                    records.append(evt)
                if auto_commit and available:
                    self.consumer_offsets[consumer_group][topic][pid] = current_offset + len(available)
                if len(records) >= max_records:
                    break

            if auto_commit and records:
                self._save_offsets()
            return records

    def commit(self, consumer_group: str, topic: str, partition_id: int, offset: int) -> None:
        """Manually commits consumer group offset for a partition."""
        with self._lock:
            self.consumer_offsets[consumer_group][topic][partition_id] = offset
            self._save_offsets()

    def seek(self, consumer_group: str, topic: str, partition_id: int, offset: int) -> None:
        """Rewinds or fast-forwards consumer group offset to a specific position."""
        with self._lock:
            high_water = len(self.partitions[topic][partition_id])
            clamped = max(0, min(offset, high_water))
            self.consumer_offsets[consumer_group][topic][partition_id] = clamped
            self._save_offsets()

    def reset_offsets(self, topic: str, consumer_group: str, target: str = "earliest") -> None:
        """Resets all partition offsets for a consumer group to earliest (0) or latest (high watermark)."""
        with self._lock:
            for pid in range(self.num_partitions):
                if target == "earliest":
                    self.consumer_offsets[consumer_group][topic][pid] = 0
                else:
                    self.consumer_offsets[consumer_group][topic][pid] = len(self.partitions[topic][pid])
            self._save_offsets()

    def get_lag(self, topic: str, consumer_group: str) -> Dict[str, Any]:
        """Calculates current consumer lag per partition and in total."""
        with self._lock:
            total_lag = 0
            per_partition = {}
            for pid in range(self.num_partitions):
                high_water = len(self.partitions[topic][pid])
                current = self.consumer_offsets[consumer_group][topic][pid]
                lag = max(0, high_water - current)
                total_lag += lag
                per_partition[pid] = {
                    "high_water": high_water,
                    "current_offset": current,
                    "lag": lag,
                }
            return {"total_lag": total_lag, "partitions": per_partition}

    def get_partition_stats(self, topic: str) -> Dict[str, Any]:
        """Returns partition message count stats."""
        with self._lock:
            stats = {}
            total = 0
            for pid in range(self.num_partitions):
                cnt = len(self.partitions[topic][pid])
                total += cnt
                stats[pid] = cnt
            return {"topic": topic, "total_messages": total, "partitions": stats}
