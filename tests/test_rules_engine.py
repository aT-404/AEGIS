import pytest
from src.anomaly.rules_engine import RulesEngine


def test_intrusion_rule_fires_once_on_entry():
    engine = RulesEngine()
    fps = 30.0

    hit = {
        "track_id": "T-001",
        "zone_name": "R-01",
        "zone_label": "Restricted Zone",
        "bbox": [100, 100, 150, 150],
        "centroid": [125, 125],
        "rules": {"intrusion": True},
    }

    # Frame 0: first entry -> fires INTRUSION
    events_0 = engine.evaluate([hit], frame_idx=0, fps=fps)
    assert len(events_0) == 1
    assert events_0[0]["anomaly_type"] == "INTRUSION"
    assert events_0[0]["track_id"] == "T-001"

    # Frame 1: continuous presence -> must NOT fire another INTRUSION event
    events_1 = engine.evaluate([hit], frame_idx=1, fps=fps)
    assert len(events_1) == 0

    # Frame 30: still inside -> no alert
    events_30 = engine.evaluate([hit], frame_idx=30, fps=fps)
    assert len(events_30) == 0


def test_loitering_rule_fires_exactly_once_at_threshold():
    engine = RulesEngine(departure_grace_frames=5)
    fps = 30.0
    threshold_seconds = 10.0  # 300 frames

    hit = {
        "track_id": "T-002",
        "zone_name": "R-02",
        "zone_label": "High-Voltage Area",
        "bbox": [200, 200, 250, 250],
        "centroid": [225, 225],
        "rules": {"intrusion": False, "loitering_threshold_seconds": threshold_seconds},
    }

    # Simulate frames 0 to 299 (dwell < 10s)
    loitering_alerts_before = []
    for f in range(300):
        evs = engine.evaluate([hit], frame_idx=f, fps=fps)
        loitering_alerts_before.extend([e for e in evs if e["anomaly_type"] == "LOITERING"])
    assert len(loitering_alerts_before) == 0

    # Frame 300: dwell reaches exactly 10.0 seconds -> must fire
    evs_300 = engine.evaluate([hit], frame_idx=300, fps=fps)
    loitering_at_300 = [e for e in evs_300 if e["anomaly_type"] == "LOITERING"]
    assert len(loitering_at_300) == 1
    assert loitering_at_300[0]["dwell_seconds"] >= 10.0

    # Frames 301 to 450: track still loitering -> must NOT fire repeated alerts
    subsequent_loitering_alerts = []
    for f in range(301, 451):
        evs = engine.evaluate([hit], frame_idx=f, fps=fps)
        subsequent_loitering_alerts.extend([e for e in evs if e["anomaly_type"] == "LOITERING"])
    assert len(subsequent_loitering_alerts) == 0


def test_loitering_jitter_tolerance():
    engine = RulesEngine(departure_grace_frames=5)
    fps = 30.0
    threshold = 5.0  # 150 frames

    hit = {
        "track_id": "T-003",
        "zone_name": "R-01",
        "zone_label": "Area",
        "bbox": [10, 10, 20, 20],
        "centroid": [15, 15],
        "rules": {"intrusion": False, "loitering_threshold_seconds": threshold},
    }

    # Present frames 0-100
    for f in range(101):
        engine.evaluate([hit], frame_idx=f, fps=fps)

    # Frame 101, 102, 103 (3 frames absent, < grace of 5)
    for f in [101, 102, 103]:
        engine.evaluate([], frame_idx=f, fps=fps)

    # Returns at frame 104 -> dwell timer should NOT have reset to 0
    engine.evaluate([hit], frame_idx=104, fps=fps)

    # At frame 150 (total time = 5.0s), loitering should fire
    evs_150 = engine.evaluate([hit], frame_idx=150, fps=fps)
    assert any(e["anomaly_type"] == "LOITERING" for e in evs_150)


def test_crowd_rule():
    engine = RulesEngine(crowd_cooldown_seconds=1.0)
    fps = 30.0

    def make_hit(tid):
        return {
            "track_id": tid,
            "zone_name": "GATE-01",
            "zone_label": "Gate",
            "bbox": [10, 10, 20, 20],
            "centroid": [15, 15],
            "rules": {"crowd_threshold": 2},
        }

    # 2 tracks: threshold is 2 -> not exceeded
    hits_2 = [make_hit("T-1"), make_hit("T-2")]
    evs_frame0 = engine.evaluate(hits_2, frame_idx=0, fps=fps)
    assert not any(e["anomaly_type"] == "CROWD" for e in evs_frame0)

    # 3 tracks: threshold is 2 -> exceeded! Crowd event emitted
    hits_3 = [make_hit("T-1"), make_hit("T-2"), make_hit("T-3")]
    evs_frame1 = engine.evaluate(hits_3, frame_idx=1, fps=fps)
    crowd_evs = [e for e in evs_frame1 if e["anomaly_type"] == "CROWD"]
    assert len(crowd_evs) == 1
    assert crowd_evs[0]["zone_count"] == 3

    # Frame 5 (0.16s later, within 1.0s cooldown): should not emit another crowd alert
    evs_frame5 = engine.evaluate(hits_3, frame_idx=5, fps=fps)
    assert not any(e["anomaly_type"] == "CROWD" for e in evs_frame5)
