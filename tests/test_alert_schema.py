import json
import numpy as np
import pytest
from pydantic import ValidationError
from src.alerts.alert_generator import Alert, AlertGenerator


def test_alert_schema_valid():
    valid_data = {
        "alert_id": "ALERT-0001",
        "timestamp": "2026-10-26T12:00:00Z",
        "source": "factory_floor_sample.mp4",
        "frame_number": 120,
        "track_id": "T-005",
        "anomaly_type": "INTRUSION",
        "zone_name": "R-01",
        "zone_label": "Machinery Exclusion Zone",
        "confidence": 0.89,
        "bbox_at_trigger": [120, 250, 200, 400],
        "dwell_seconds": 0.0,
        "zone_count": 1,
    }
    alert = Alert(**valid_data)
    assert alert.alert_id == "ALERT-0001"
    assert alert.anomaly_type == "INTRUSION"
    assert alert.bbox_at_trigger == [120, 250, 200, 400]


def test_alert_schema_invalid_anomaly_type():
    invalid_data = {
        "alert_id": "ALERT-0002",
        "timestamp": "2026-10-26T12:00:00Z",
        "source": "test.mp4",
        "frame_number": 10,
        "track_id": "T-001",
        "anomaly_type": "INVALID_ANOMALY",  # Not in Literal["INTRUSION", "LOITERING", "CROWD"]
        "zone_name": "R-01",
        "zone_label": "Test Zone",
        "bbox_at_trigger": [0, 0, 10, 10],
    }
    with pytest.raises(ValidationError):
        Alert(**invalid_data)


def test_alert_generator_lifecycle(tmp_path):
    generator = AlertGenerator(output_dir=str(tmp_path), source="unit_test_stream")
    event = {
        "anomaly_type": "LOITERING",
        "track_id": "T-042",
        "zone_name": "R-02",
        "zone_label": "High-Voltage Area",
        "confidence": 0.94,
        "bbox": [50, 50, 100, 100],
        "centroid": [75, 75],
        "frame_number": 450,
        "dwell_seconds": 32.5,
    }

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    alert = generator.generate(event, frame=dummy_frame)

    assert alert.alert_id == "ALERT-0001"
    json_path = tmp_path / "ALERT-0001.json"
    thumb_path = tmp_path / "frames" / "ALERT-0001.jpg"

    assert json_path.exists()
    assert thumb_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["alert_id"] == "ALERT-0001"
    assert loaded["anomaly_type"] == "LOITERING"
    assert loaded["dwell_seconds"] == 32.5

    # Test counter increments
    second_alert = generator.generate(event)
    assert second_alert.alert_id == "ALERT-0002"
