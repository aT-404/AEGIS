# AEGIS Architecture

AEGIS (Adaptive Edge-Guided Intelligence for Surveillance) is designed to operate on edge devices with minimal latency, running real-time video analytics without requiring cloud round-trips or dedicated GPUs.

```
flowchart TD
    A["Video Source (RTSP Stream / MP4 File)"] --> B["Stream Reader (src/ingestion/stream_reader.py)"]
    B --> C["ONNX Runtime YOLO Detector (src/detection/yolo_detector.py)"]
    C --> D["ByteTrack Multi-Object Tracker (src/tracking/byte_tracker.py)"]
    D --> E["Zone Monitor — Point-in-Polygon (src/anomaly/zone_monitor.py)"]
    E --> F["Anomaly Rules Engine (src/anomaly/rules_engine.py)"]
    F --> G["Alert Generator (src/alerts/alert_generator.py)"]
    G --> H["Alert JSON Files (alerts/)"]
    G --> I["Alert Queue — In-Memory"]
    I --> J["Streamlit Dashboard (dashboard/app.py)"]
    H --> J
```

## Pipeline Stages

### 1. Ingestion (`src/ingestion/stream_reader.py`)
Decouples frame retrieval from downstream inference. Supports IP camera RTSP streams, local MP4/AVI files, and webcam indices. Yields frame indices and raw frames as numpy arrays while maintaining frame rate synchronization and handling connection drops.

### 2. Detection (`src/detection/yolo_detector.py`)
Runs YOLOv8n (nano variant, 3.2M parameters) served via ONNX Runtime using CPU execution providers. Preprocessing resizes and normalizes incoming frames to 640×640. Postprocessing performs non-maximum suppression (NMS) and maps bounding boxes back to the original camera coordinates.

### 3. Tracking (`src/tracking/byte_tracker.py`)
ByteTrack associates detections across frames by matching bounding box intersection-over-union (IoU) with a Kalman filter motion predictor. Unlike ReID-based approaches (such as DeepSORT), ByteTrack does not execute a secondary deep embedding network, ensuring high frame rates on standard CPU hardware.

### 4. Zone Monitoring (`src/anomaly/zone_monitor.py`)
Computes point-in-polygon containment using Shapely geometries. Each object centroid is checked against user-defined zones configured in `config/zones.yaml`. Also renders semi-transparent polygon overlays and zone boundary lines for visual verification.

### 5. Rules Engine (`src/anomaly/rules_engine.py`)
Stateful anomaly evaluation:
- **Intrusion:** Triggered immediately when a tracked entity's centroid enters an unauthorized zone.
- **Loitering:** Edge-triggered when an entity's dwell time exceeds the configured threshold. Includes a 5-frame jitter tolerance to prevent false resets from bounding box noise.
- **Crowd Formation:** Evaluates instantaneous zone occupancy with a 1-second hysteresis cooldown to avoid alert flooding.

### 6. Alert Generator (`src/alerts/alert_generator.py`)
Validates events against a strict Pydantic `Alert` schema, saves timestamped JSON payloads (`alerts/ALERT-XXXX.json`), captures evidence frame thumbnails (`alerts/frames/ALERT-XXXX.jpg`), and puts events into an in-memory queue for dashboard consumption.

### 7. Dashboard (`dashboard/app.py`)
Streamlit-based monitoring console displaying the annotated feed, real-time alert logs with evidence thumbnails, per-zone occupancy counters, and historical alert analytics.
