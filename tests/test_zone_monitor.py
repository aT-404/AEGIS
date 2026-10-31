import pytest
from src.anomaly.zone_monitor import ZoneMonitor


def test_zone_monitor_containment():
    # Define a 100x100 square zone from (100, 100) to (200, 200)
    zones = [
        {
            "name": "TEST-ZONE-01",
            "label": "Test Restricted Area",
            "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            "rules": {"intrusion": True, "loitering_threshold_seconds": 10},
        }
    ]
    monitor = ZoneMonitor(zones)

    tracks = [
        # Inside
        {"track_id": "T-001", "centroid": [150, 150], "bbox": [140, 140, 160, 160]},
        {"track_id": "T-002", "centroid": [110, 110], "bbox": [105, 105, 115, 115]},
        {"track_id": "T-003", "centroid": [190, 190], "bbox": [185, 185, 195, 195]},
        # Outside
        {"track_id": "T-004", "centroid": [50, 50], "bbox": [40, 40, 60, 60]},
        {"track_id": "T-005", "centroid": [250, 250], "bbox": [240, 240, 260, 260]},
        {"track_id": "T-006", "centroid": [150, 50], "bbox": [140, 40, 160, 60]},
        {"track_id": "T-007", "centroid": [50, 150], "bbox": [40, 140, 60, 160]},
        {"track_id": "T-008", "centroid": [300, 150], "bbox": [290, 140, 310, 160]},
        {"track_id": "T-009", "centroid": [150, 300], "bbox": [140, 290, 160, 310]},
        # Boundary / edge point (100, 150)
        {"track_id": "T-010", "centroid": [100, 150], "bbox": [95, 145, 105, 155]},
    ]

    hits = monitor.check_tracks(tracks)
    hit_ids = {h["track_id"] for h in hits}

    # T-001, T-002, T-003 are inside; T-010 is on boundary
    assert "T-001" in hit_ids
    assert "T-002" in hit_ids
    assert "T-003" in hit_ids
    assert "T-010" in hit_ids

    # Outside tracks must NOT be in hits
    assert "T-004" not in hit_ids
    assert "T-005" not in hit_ids
    assert "T-006" not in hit_ids
    assert "T-007" not in hit_ids
    assert "T-008" not in hit_ids
    assert "T-009" not in hit_ids
    assert len(hits) == 4
