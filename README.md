# AEGIS — Adaptive Edge-Guided Intelligence for Surveillance

A real-time video analytics system for industrial environments. AEGIS processes live RTSP streams or recorded video, runs multi-object detection and tracking using a YOLO + ByteTrack pipeline, evaluates configurable anomaly rules (zone intrusion, loitering, crowd formation), and emits structured alert JSON — all optimized to run on edge hardware without a GPU.

Built for the operational gap that exists in industrial surveillance: hundreds of cameras produce continuous video, but centralized cloud analytics creates latency (a 2-second round-trip is too slow for safety-critical zones), bandwidth cost, and data privacy exposure. AEGIS runs the inference at the edge and sends only alert events to a central dashboard.

---

## Key Features

- **YOLOv8n Detection:** Uses the nano variant of YOLOv8 — 3.2M parameters, 8.7ms inference on CPU — tuned specifically for pedestrian and equipment detection in industrial footage. ONNX-exported for runtime portability.
- **ByteTrack Multi-Object Tracking:** Maintains consistent object identities across frames using a motion-model-based association algorithm. Lighter than DeepSORT — no re-identification feature extraction — which makes it practical on CPU-only edge hardware.
- **YAML-Configured ROI Zones:** Define restricted zones as polygons in `config/zones.yaml`. No code changes required to add, remove, or reshape zones — redeploy config, restart stream.
- **Rules Engine (Three Anomaly Types):**
  - **Zone Intrusion:** Any tracked object enters a defined restricted polygon.
  - **Loitering:** A tracked object remains inside a zone for longer than a configurable dwell threshold (default: 30 seconds).
  - **Crowd Formation:** The count of simultaneous tracked objects inside a zone exceeds a configured limit.
- **Structured Alert JSON:** Every triggered anomaly produces a fully typed alert record with timestamp, track ID, anomaly type, zone name, confidence, and the bounding box at the moment of trigger — ready to log or forward to a SIEM or operations dashboard.
- **ONNX Runtime Inference:** YOLOv8n exported to ONNX and served via ONNX Runtime, achieving consistent sub-20ms per-frame inference on a standard laptop CPU. No CUDA dependency.
- **Streamlit Monitoring Dashboard:** Live frame display with bounding boxes and zone overlays, real-time alert log, per-zone occupancy counts, and historical alert statistics.

---

## Architecture Diagram

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

---

## Alert Walkthrough: `ALERT-0031` (Loitering in Restricted Zone R-02)

Scenario: A ground-level camera on a factory floor monitors Zone R-02, a high-voltage equipment area marked off-limits to unauthorized personnel.

1. **Frame 1842 (00:01:01.4):** YOLOv8n detects a person, confidence 0.87, bbox `[412, 318, 531, 608]`.
2. **Frames 1842–1984 (00:01:01.4 → 00:01:06.1):** ByteTrack assigns Track ID `T-017`. Zone monitor confirms the centroid `[471, 463]` is inside the R-02 polygon for 32 consecutive seconds.
3. **Loitering Rule fires:** Dwell threshold is 30 seconds. T-017 has exceeded it.
4. **Alert generated:**
```json
{
  "alert_id": "ALERT-0031",
  "timestamp": "2026-11-08T14:23:07Z",
  "track_id": "T-017",
  "anomaly_type": "LOITERING",
  "zone_name": "R-02",
  "zone_label": "High-Voltage Equipment Area",
  "dwell_seconds": 32,
  "confidence": 0.87,
  "bbox_at_trigger": [412, 318, 531, 608],
  "frame_number": 1984,
  "source": "factory_floor_cam3.mp4"
}
```
5. **Dashboard:** Alert appears in the live log. Zone R-02 count shows 1 occupant. Frame thumbnail is saved to `alerts/frames/ALERT-0031.jpg`.

---

## Quick Start & Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configure Zones

Edit `config/zones.yaml` to define your restricted zones as pixel-coordinate polygons:

```yaml
zones:
  - name: R-02
    label: "High-Voltage Equipment Area"
    polygon: [[310, 200], [640, 200], [640, 580], [310, 580]]
    rules:
      intrusion: true
      loitering_threshold_seconds: 30
      crowd_threshold: 2
```

### 3. Launch Dashboard (with video source)

```bash
streamlit run dashboard/app.py -- --source data/samples/factory_floor_sample.mp4
```

### 4. Run Pipeline via CLI

```bash
python src/main.py --source data/samples/factory_floor_sample.mp4 --output alerts/
```

### 5. Export Model to ONNX (Edge Deployment)

```bash
python src/optimization/model_export.py --model yolov8n.pt --output data/models/yolov8n.onnx
```

### 6. Run Test Suite

```bash
pytest
```

---

## Repository Structure

```
AEGIS/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── IMPLEMENTATION_PLAN.md
├── PROJECT_JOURNAL.md
├── config/
│   ├── zones.yaml
│   ├── rules.yaml
│   └── tracker.yaml
├── data/
│   ├── samples/           ← sample video clips for testing
│   └── models/            ← YOLO weights + ONNX model
├── src/
│   ├── ingestion/
│   │   └── stream_reader.py
│   ├── detection/
│   │   └── yolo_detector.py
│   ├── tracking/
│   │   └── byte_tracker.py
│   ├── anomaly/
│   │   ├── zone_monitor.py
│   │   └── rules_engine.py
│   ├── alerts/
│   │   └── alert_generator.py
│   ├── optimization/
│   │   └── model_export.py
│   └── main.py
├── alerts/
│   └── frames/            ← frame thumbnails at alert trigger
├── dashboard/
│   └── app.py
├── docs/
│   ├── architecture.md
│   ├── detection-methodology.md
│   ├── edge-optimization.md
│   └── setup.md
├── tests/
│   ├── test_zone_monitor.py
│   ├── test_rules_engine.py
│   └── test_alert_schema.py
└── .github/
    └── workflows/
        └── tests.yml
```

---

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Detection & Tracking Methodology](docs/detection-methodology.md)
- [Edge Optimization Guide](docs/edge-optimization.md)
- [Setup Guide](docs/setup.md)

---

## Disclaimer

Zone definitions use pixel coordinates and are camera-specific. Intrusion detection accuracy depends on camera angle, resolution, and scene complexity. The confidence thresholds in `config/rules.yaml` are tuned on the sample dataset — production deployment requires calibration on real footage from each camera location.
