from collections import defaultdict
from typing import List, Dict, Any, Optional, Set


class RulesEngine:
    """Stateful anomaly detection engine for industrial surveillance.

    Rule defaults can be overridden at construction time by passing a
    ``rules_cfg`` dict loaded from ``config/rules.yaml``.  Keys that are
    absent in the dict fall back to the built-in defaults shown below.

    Example config/rules.yaml structure::

        intrusion:
          enabled: true
          min_confidence: 0.4
        loitering:
          enabled: true
          threshold_seconds: 30
          check_interval_frames: 30
        crowd:
          enabled: true
          max_count: 3
    """

    def __init__(
        self,
        departure_grace_frames: int = 5,
        crowd_cooldown_seconds: float = 1.0,
        rules_cfg: Optional[Dict[str, Any]] = None,
    ):
        # dwell_start[track_id][zone_name] = timestamp in seconds of first entry
        self._dwell_start: Dict[str, Dict[str, float]] = defaultdict(dict)
        # intrusion_fired[track_id][zone_name] = bool
        self._intrusion_fired: Dict[str, Dict[str, bool]] = defaultdict(dict)
        # loitering_fired[track_id][zone_name] = bool
        self._loitering_fired: Dict[str, Dict[str, bool]] = defaultdict(dict)
        # tracks_last_seen[track_id][zone_name] = frame_number
        self._tracks_last_seen: Dict[str, Dict[str, int]] = defaultdict(dict)
        # zone_occupancy[zone_name] = set of track_ids currently active
        self._zone_occupancy: Dict[str, Set[str]] = defaultdict(set)
        # last_crowd_alert_time[zone_name] = timestamp in seconds
        self._last_crowd_alert_time: Dict[str, float] = {}

        self.departure_grace_frames = departure_grace_frames
        self.crowd_cooldown_seconds = crowd_cooldown_seconds

        # --- Rule configuration (from rules.yaml or built-in defaults) ---
        cfg = rules_cfg or {}

        intr = cfg.get("intrusion", {})
        self.intrusion_enabled: bool = bool(intr.get("enabled", True))
        self.intrusion_min_confidence: float = float(intr.get("min_confidence", 0.4))

        loit = cfg.get("loitering", {})
        self.loitering_enabled: bool = bool(loit.get("enabled", True))
        self.loitering_threshold_seconds: float = float(loit.get("threshold_seconds", 30))
        self.loitering_check_interval_frames: int = int(loit.get("check_interval_frames", 30))

        crowd = cfg.get("crowd", {})
        self.crowd_enabled: bool = bool(crowd.get("enabled", True))
        self.crowd_max_count: int = int(crowd.get("max_count", 3))

    def evaluate(self, zone_hits: List[Dict[str, Any]], frame_idx: int, fps: float) -> List[Dict[str, Any]]:
        """
        Evaluates zone hits for the current frame against active rules.
        zone_hits: output of ZoneMonitor.check_tracks()
        """
        current_time = frame_idx / fps
        triggered = []

        # Map current frame hits by zone
        current_zone_tracks: Dict[str, Set[str]] = defaultdict(set)
        current_zone_hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for hit in zone_hits:
            z_name = hit["zone_name"]
            tid = hit["track_id"]
            current_zone_tracks[z_name].add(tid)
            current_zone_hits[z_name].append(hit)
            self._tracks_last_seen[tid][z_name] = frame_idx

        # Check for departures exceeding grace period
        all_zones = set(self._zone_occupancy.keys()) | set(current_zone_tracks.keys())
        for zone_name in all_zones:
            tracked_in_zone = list(self._zone_occupancy[zone_name])
            for tid in tracked_in_zone:
                last_seen = self._tracks_last_seen.get(tid, {}).get(zone_name, -1)
                if (frame_idx - last_seen) > self.departure_grace_frames:
                    # Clean up departed track state
                    self._dwell_start[tid].pop(zone_name, None)
                    self._intrusion_fired[tid].pop(zone_name, None)
                    self._loitering_fired[tid].pop(zone_name, None)
                    self._tracks_last_seen[tid].pop(zone_name, None)
                    self._zone_occupancy[zone_name].discard(tid)

        # Update active occupancy sets
        for zone_name, tids in current_zone_tracks.items():
            self._zone_occupancy[zone_name].update(tids)

        # Evaluate individual track rules: Intrusion & Loitering
        for hit in zone_hits:
            track_id = hit["track_id"]
            zone_name = hit["zone_name"]
            rules = hit.get("rules", {})

            # Rule 1 — Zone Intrusion
            confidence = hit.get("confidence", 1.0)
            if (
                self.intrusion_enabled
                and rules.get("intrusion", True)
                and confidence >= self.intrusion_min_confidence
            ):
                if not self._intrusion_fired[track_id].get(zone_name, False):
                    self._intrusion_fired[track_id][zone_name] = True
                    triggered.append(
                        self._make_event("INTRUSION", hit, frame_idx, dwell_seconds=0.0)
                    )

            # Record dwell start time if new entry
            if zone_name not in self._dwell_start[track_id]:
                self._dwell_start[track_id][zone_name] = current_time

            dwell = current_time - self._dwell_start[track_id][zone_name]

            # Rule 2 — Loitering
            loiter_thresh = rules.get("loitering_threshold_seconds", self.loitering_threshold_seconds)
            if self.loitering_enabled and dwell >= loiter_thresh:
                # Trigger exactly once per loitering episode
                if not self._loitering_fired[track_id].get(zone_name, False):
                    self._loitering_fired[track_id][zone_name] = True
                    triggered.append(
                        self._make_event(
                            "LOITERING",
                            hit,
                            frame_idx,
                            dwell_seconds=round(dwell, 2),
                        )
                    )

        # Rule 3 — Crowd Formation (evaluated per zone)
        for zone_name, hits in current_zone_hits.items():
            if not hits or not self.crowd_enabled:
                continue
            rules = hits[0].get("rules", {})
            crowd_thresh = rules.get("crowd_threshold", self.crowd_max_count)
            if crowd_thresh is not None:
                current_count = len(self._zone_occupancy[zone_name])
                if current_count > crowd_thresh:
                    last_alert = self._last_crowd_alert_time.get(zone_name, -999.0)
                    if (current_time - last_alert) >= self.crowd_cooldown_seconds:
                        self._last_crowd_alert_time[zone_name] = current_time
                        # Emit crowd event for the zone
                        representative_hit = hits[0]
                        triggered.append(
                            self._make_event(
                                "CROWD",
                                representative_hit,
                                frame_idx,
                                zone_count=current_count,
                            )
                        )

        return triggered

    def _make_event(self, anomaly_type: str, hit: Dict[str, Any], frame_idx: int, **kwargs) -> Dict[str, Any]:
        return {
            "anomaly_type": anomaly_type,
            "track_id": hit["track_id"],
            "zone_name": hit["zone_name"],
            "zone_label": hit["zone_label"],
            "bbox": hit["bbox"],
            "centroid": hit["centroid"],
            "confidence": hit.get("confidence", 1.0),
            "frame_number": frame_idx,
            **kwargs,
        }
