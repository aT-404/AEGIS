import json
import os
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
import cv2
from pydantic import BaseModel, Field


class Alert(BaseModel):
    alert_id: str = Field(description="Unique incremental alert identifier, e.g., ALERT-0001")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of trigger")
    source: str = Field(description="Video stream identifier or filename")
    frame_number: int = Field(description="Video frame index when the alert triggered")
    track_id: str = Field(description="Track ID of the associated entity")
    anomaly_type: Literal["INTRUSION", "LOITERING", "CROWD"] = Field(description="Classification of anomaly")
    zone_name: str = Field(description="Identifier of the restricted zone")
    zone_label: str = Field(description="Descriptive label of the zone")
    confidence: Optional[float] = Field(default=1.0, description="Detection confidence score")
    bbox_at_trigger: List[int] = Field(description="Bounding box [x1, y1, x2, y2] at trigger time")
    dwell_seconds: Optional[float] = Field(default=None, description="Time in seconds entity remained in zone")
    zone_count: Optional[int] = Field(default=None, description="Concurrent occupants in zone if crowd alert")


class AlertGenerator:
    """Manages alert record persistence, evidence frame capture, and streaming dispatch."""

    def __init__(self, output_dir: str = "alerts/", source: str = "stream"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)

        self._source = source
        self._counter = self._load_counter()
        self._queue: queue.Queue = queue.Queue()

    def _load_counter(self) -> int:
        existing = list(self._output_dir.glob("ALERT-*.json"))
        if not existing:
            return 0
        nums = []
        for p in existing:
            try:
                parts = p.stem.split("-")
                if len(parts) >= 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            except Exception:
                continue
        return max(nums) if nums else 0

    def generate(self, event: dict, frame: Optional[cv2.Mat] = None) -> Alert:
        """Constructs, validates, writes to disk, and enqueues an alert record."""
        self._counter += 1
        alert_id = f"ALERT-{self._counter:04d}"

        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self._source,
            frame_number=event["frame_number"],
            track_id=event["track_id"],
            anomaly_type=event["anomaly_type"],
            zone_name=event["zone_name"],
            zone_label=event["zone_label"],
            confidence=event.get("confidence", 1.0),
            bbox_at_trigger=event["bbox"],
            dwell_seconds=event.get("dwell_seconds"),
            zone_count=event.get("zone_count"),
        )

        # Write JSON alert payload
        out_path = self._output_dir / f"{alert_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(alert.model_dump(), f, indent=2)

        # Save annotated evidence frame thumbnail
        if frame is not None:
            thumb_path = self._frames_dir / f"{alert_id}.jpg"
            evidence_frame = frame.copy()
            x1, y1, x2, y2 = alert.bbox_at_trigger
            # Draw highlight box around trigger object
            cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            header = f"{alert.alert_id}: {alert.anomaly_type} [{alert.zone_name}]"
            cv2.putText(
                evidence_frame,
                header,
                (max(x1, 10), max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(thumb_path), evidence_frame)

        # Push to in-memory queue
        self._queue.put(alert.model_dump())
        return alert

    def get_queue(self) -> queue.Queue:
        return self._queue
