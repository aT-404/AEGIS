import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Any
from shapely.geometry import Point, Polygon


def load_zones(config_path: str = "config/zones.yaml") -> List[Dict[str, Any]]:
    """Loads zone configurations from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("zones", []) if data else []


class ZoneMonitor:
    """Monitors track centroids against user-defined polygon regions of interest."""

    def __init__(self, zones: List[Dict[str, Any]]):
        self._zones = []
        for z in zones:
            poly_coords = z["polygon"]
            # Ensure polygon is closed or has at least 3 points
            if len(poly_coords) >= 3:
                self._zones.append({
                    "name": z["name"],
                    "label": z.get("label", z["name"]),
                    "polygon": Polygon(poly_coords),
                    "raw_polygon": poly_coords,
                    "rules": z.get("rules", {}),
                })

    def check_tracks(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For each track, checks which zones its centroid is inside (or on the boundary).
        Returns list of match dictionaries:
        {track_id, zone_name, zone_label, rules, bbox, centroid}
        """
        results = []
        for track in tracks:
            cx, cy = track["centroid"]
            point = Point(cx, cy)
            for zone in self._zones:
                # contains or touches (for boundary edge cases)
                if zone["polygon"].contains(point) or zone["polygon"].touches(point):
                    results.append({
                        "track_id": track["track_id"],
                        "zone_name": zone["name"],
                        "zone_label": zone["label"],
                        "rules": zone["rules"],
                        "bbox": track["bbox"],
                        "centroid": track["centroid"],
                        "confidence": track.get("confidence", 1.0),
                    })
        return results

    def draw_zones(self, frame: np.ndarray, color=(0, 0, 255), alpha=0.25) -> np.ndarray:
        """Draws semi-transparent zone overlays and boundary lines on the frame."""
        overlay = frame.copy()
        for zone in self._zones:
            pts = np.array(zone["raw_polygon"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Zone label placement
            min_x = int(min(p[0] for p in zone["raw_polygon"]))
            min_y = int(min(p[1] for p in zone["raw_polygon"]))
            label_text = f"ZONE: {zone['name']} ({zone['label']})"
            cv2.putText(
                frame,
                label_text,
                (min_x + 8, max(min_y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label_text,
                (min_x + 8, max(min_y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        return frame
