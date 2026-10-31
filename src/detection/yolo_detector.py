import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class YOLODetector:
    """Unified YOLO detector supporting both ONNX Runtime (edge optimized) and PyTorch (.pt)."""

    def __init__(self, model_path: str = "data/models/yolov8n.onnx", conf_threshold: float = 0.4, iou_threshold: float = 0.5):
        self.model_path = str(model_path)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.input_size = 640
        self.is_onnx = self.model_path.lower().endswith(".onnx")

        if self.is_onnx:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Match available physical CPU cores for edge responsiveness
            num_cores = max(1, min(os.cpu_count() or 4, 8))
            sess_options.intra_op_num_threads = num_cores
            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self.names = {i: name for i, name in enumerate(COCO_CLASSES)}
        else:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self.names = self._model.names

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.expand_dims(normalized.transpose(2, 0, 1), axis=0)

    def _postprocess_onnx(self, output: np.ndarray, orig_shape: tuple) -> List[Dict[str, Any]]:
        # YOLOv8 ONNX output shape: [1, 84, 8400] -> transpose to [8400, 84]
        pred = output[0][0].T  # shape: [8400, 84]

        # Vectorized extraction of boxes and class scores
        boxes_raw = pred[:, :4]
        scores_matrix = pred[:, 4:]

        class_ids = np.argmax(scores_matrix, axis=1)
        confidences = scores_matrix[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf
        if not np.any(mask):
            return []

        filt_boxes = boxes_raw[mask]
        filt_conf = confidences[mask]
        filt_cids = class_ids[mask]

        oh, ow = orig_shape[:2]
        scale_x = ow / self.input_size
        scale_y = oh / self.input_size

        cx = filt_boxes[:, 0]
        cy = filt_boxes[:, 1]
        w = filt_boxes[:, 2]
        h = filt_boxes[:, 3]

        x1 = np.clip((cx - w / 2.0) * scale_x, 0, ow - 1).astype(int)
        y1 = np.clip((cy - h / 2.0) * scale_y, 0, oh - 1).astype(int)
        x2 = np.clip((cx + w / 2.0) * scale_x, 0, ow - 1).astype(int)
        y2 = np.clip((cy + h / 2.0) * scale_y, 0, oh - 1).astype(int)

        bw = x2 - x1
        bh = y2 - y1
        valid_size = (bw > 0) & (bh > 0)

        boxes = []
        scores = []
        cids = []
        for i in np.where(valid_size)[0]:
            boxes.append([int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])])
            scores.append(float(filt_conf[i]))
            cids.append(int(filt_cids[i]))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        results = []
        if len(indices) > 0:
            indices_flat = indices.flatten() if hasattr(indices, "flatten") else indices
            for idx in indices_flat:
                i = int(idx)
                bx, by, bw_i, bh_i = boxes[i]
                cid = cids[i]
                results.append({
                    "class_id": cid,
                    "class_name": self.names.get(cid, str(cid)),
                    "confidence": scores[i],
                    "bbox": [bx, by, bx + bw_i, by + bh_i],
                })
        return results

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Performs object detection on a single video frame."""
        if self.is_onnx:
            blob = self._preprocess(frame)
            output = self._session.run(None, {self._input_name: blob})
            return self._postprocess_onnx(output, frame.shape)
        else:
            results = self._model(frame, conf=self.conf, iou=self.iou, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cid = int(box.cls)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "class_id": cid,
                        "class_name": self._model.names.get(cid, str(cid)),
                        "confidence": float(box.conf),
                        "bbox": [x1, y1, x2, y2],
                    })
            return detections
