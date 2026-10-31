# AEGIS Setup & Calibration Guide

## Prerequisites

- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Windows, Linux, or macOS

## Installation

```bash
git clone https://github.com/your-org/AEGIS.git
cd AEGIS
pip install -r requirements.txt
```

## Model Preparation

1. Download the base YOLOv8n weights and export to ONNX:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Move or export to data/models
python src/optimization/model_export.py --model yolov8n.pt --output data/models/yolov8n.onnx
```

## Calibrating Zone Coordinates

To calibrate polygon coordinates for a new camera feed:

1. Capture a reference frame from the stream using OpenCV.
2. Record the pixel coordinates `[x, y]` of the zone vertices in `config/zones.yaml`.
3. Specify the desired anomaly rules:
   - `intrusion: true` (alert on entry)
   - `loitering_threshold_seconds: 30` (alert if track dwell time exceeds N seconds)
   - `crowd_threshold: 3` (alert if concurrent tracks in zone exceed count)

## Running AEGIS

### Command-line Pipeline
```bash
python src/main.py --source data/samples/factory_floor_sample.mp4 --write-video
```

### Streamlit Interactive Console
```bash
streamlit run dashboard/app.py
```
