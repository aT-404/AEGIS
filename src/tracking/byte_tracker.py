import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes IoU matrix between N boxes_a and M boxes_b: [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.ascontiguousarray(boxes_a, dtype=np.float32)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T),
    )
    ih = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T),
    )

    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-6)
    return inter / union


class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes in image space."""

    count = 0

    def __init__(self, bbox: List[int], score: float = 1.0, class_name: str = "person"):
        # State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
        self.kf_dim_x = 8
        self.kf_dim_z = 4

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=np.float32)

        # Covariance matrices
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.01]).astype(np.float32)

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[:4] = self._bbox_to_z(bbox)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.score = score
        self.class_name = class_name
        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.is_activated = False

    def _bbox_to_z(self, bbox: List[int]) -> np.ndarray:
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([[x], [y], [r], [h]], dtype=np.float32)

    def _z_to_bbox(self, x: np.ndarray) -> List[int]:
        w = float(x[2, 0] * x[3, 0])
        h = float(x[3, 0])
        x1 = int(round(x[0, 0] - w / 2.0))
        y1 = int(round(x[1, 0] - h / 2.0))
        x2 = int(round(x[0, 0] + w / 2.0))
        y2 = int(round(x[1, 0] + h / 2.0))
        return [x1, y1, x2, y2]

    def predict(self) -> List[int]:
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox: List[int], score: float, class_name: Optional[str] = None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        if class_name:
            self.class_name = class_name

        z = self._bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.kf_dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> List[int]:
        return self._z_to_bbox(self.x)


class ObjectTracker:
    """ByteTrack multi-object tracker associating high and low confidence detections."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.track_thresh = float(cfg.get("track_thresh", 0.5))
        self.match_thresh = float(cfg.get("match_thresh", 0.8))
        self.track_buffer = int(cfg.get("track_buffer", 30))
        self.frame_rate = int(cfg.get("frame_rate", 30))
        self.min_hits = int(cfg.get("min_hits", 3))

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]], frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Updates trackers with current frame detections.
        Returns active tracks:
        [{track_id, class_name, confidence, bbox: [x1, y1, x2, y2], centroid: [cx, cy]}]
        """
        self.frame_count += 1

        # 1. Predict state for existing trackers
        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_boxes.append(pred)

        # Separate detections into D_high and D_low
        high_dets = []
        low_dets = []
        for d in detections:
            if d["confidence"] >= self.track_thresh:
                high_dets.append(d)
            elif d["confidence"] >= 0.1:
                low_dets.append(d)

        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_high_dets = list(range(len(high_dets)))

        # 2. First association with high confidence detections
        if len(self.trackers) > 0 and len(high_dets) > 0:
            track_boxes = np.array([t.get_state() for t in self.trackers], dtype=np.float32)
            det_boxes = np.array([d["bbox"] for d in high_dets], dtype=np.float32)
            iou_mat = compute_iou_matrix(track_boxes, det_boxes)

            cost_mat = 1.0 - iou_mat
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                # IoU threshold check: cost_mat[r, c] <= (1.0 - (1.0 - match_thresh)) => IoU >= (1 - match_thresh)
                # match_thresh in config is matching distance threshold (e.g. 0.8 => min IoU = 0.2)
                min_iou = 1.0 - self.match_thresh
                if iou_mat[r, c] >= min_iou:
                    self.trackers[r].update(
                        high_dets[c]["bbox"],
                        high_dets[c]["confidence"],
                        high_dets[c].get("class_name", "person"),
                    )
                    matched_tracks.add(r)
                    matched_dets.add(c)

            unmatched_trackers = [i for i in range(len(self.trackers)) if i not in matched_tracks]
            unmatched_high_dets = [j for j in range(len(high_dets)) if j not in matched_dets]

        # 3. Second association: remaining trackers with low confidence detections
        if len(unmatched_trackers) > 0 and len(low_dets) > 0:
            rem_track_boxes = np.array([self.trackers[i].get_state() for i in unmatched_trackers], dtype=np.float32)
            low_det_boxes = np.array([d["bbox"] for d in low_dets], dtype=np.float32)
            iou_mat_low = compute_iou_matrix(rem_track_boxes, low_det_boxes)

            cost_mat_low = 1.0 - iou_mat_low
            row_ind_low, col_ind_low = linear_sum_assignment(cost_mat_low)

            second_matched_tracks = set()
            for r, c in zip(row_ind_low, col_ind_low):
                if iou_mat_low[r, c] >= 0.2:
                    t_idx = unmatched_trackers[r]
                    self.trackers[t_idx].update(
                        low_dets[c]["bbox"],
                        low_dets[c]["confidence"],
                        low_dets[c].get("class_name", "person"),
                    )
                    second_matched_tracks.add(t_idx)

            unmatched_trackers = [i for i in unmatched_trackers if i not in second_matched_tracks]

        # 4. Initiate new tracks for unmatched high confidence detections
        for det_idx in unmatched_high_dets:
            d = high_dets[det_idx]
            new_tracker = KalmanBoxTracker(
                d["bbox"],
                score=d["confidence"],
                class_name=d.get("class_name", "person"),
            )
            self.trackers.append(new_tracker)

        # 5. Remove dead trackers and gather active ones
        active_results = []
        surviving_trackers = []
        for t in self.trackers:
            if t.time_since_update <= self.track_buffer:
                surviving_trackers.append(t)
                # Output active tracks that have either been seen multiple times or frame_count is early
                if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = t.get_state()
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    active_results.append({
                        "track_id": f"T-{t.id:03d}",
                        "class_name": t.class_name,
                        "confidence": float(t.score),
                        "bbox": bbox,
                        "centroid": [cx, cy],
                    })

        self.trackers = surviving_trackers
        return active_results
