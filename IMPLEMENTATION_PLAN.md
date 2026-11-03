# AEGIS — Implementation Plan

A block-by-block build record documenting the decisions, interfaces, and verification criteria for each component of the AEGIS edge surveillance pipeline.

---

## Block 0 — Project scaffold & environment

**Goal:** Establish the directory layout, dependency manifest, and CI configuration before any implementation work begins.

**Files:**
- `requirements.txt` — runtime dependencies: `ultralytics`, `onnxruntime`, `opencv-python`, `shapely`, `pydantic>=2`, `streamlit`, `pandas`, `numpy`, `scipy`, `pyyaml`, `pytest`
- `pytest.ini` — `testpaths = tests`, `python_files = test_*.py`
- `.github/workflows/tests.yml` — CI matrix on Python 3.11 and 3.12; runs `pytest`
- `.gitignore` — excludes `data/models/*.onnx`, `data/models/*.pt`, `alerts/ALERT-*.json`, `alerts/frames/*.jpg`, `data/samples/*.mp4`, `outputs/*.mp4` (large binary runtime artifacts)
- `.env.example` — `MODEL_PATH=data/models/yolov8n.onnx`

**Done when:** `pytest` discovers the test directory with zero collection errors; GitHub Actions workflow file is present and syntactically valid.

---

## Block 1 — Video ingestion

**Goal:** Abstract all video source types (MP4 file, RTSP stream, webcam device index) behind a single iterator interface.

**File:** `src/ingestion/stream_reader.py`

**Key class — `StreamReader`:**
```python
class StreamReader:
    def __init__(self, source: str):
        # source.isdigit() → cv2.VideoCapture(int(source)) for webcam
        # otherwise cv2.VideoCapture(source) for file or RTSP URL
        ...
    def frames(self) -> Generator:
        # yields (frame_idx: int, frame: np.ndarray)
        ...
    @property
    def fps(self) -> float:
        # cv2.CAP_PROP_FPS; falls back to 30.0 if unset
        ...
    @property
    def frame_size(self) -> Tuple[int, int]:
        # (width, height) from CAP_PROP_FRAME_WIDTH / HEIGHT
        ...
```

**Done when:** `StreamReader("data/samples/factory_floor_sample.mp4").frames()` yields `(0, frame)` correctly; `fps` returns a positive float; `release()` closes the capture without error.

---

## Block 2 — YOLO detection backbone

**Goal:** Run YOLOv8n inference with an ONNX Runtime primary path and an Ultralytics `.pt` fallback. The ONNX path must not import PyTorch at runtime.

**File:** `src/detection/yolo_detector.py`

**Key class — `YOLODetector`:**
```python
class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.4, iou_threshold: float = 0.5):
        self.is_onnx = model_path.lower().endswith(".onnx")
        if self.is_onnx:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = min(os.cpu_count() or 4, 8)
            self._session = ort.InferenceSession(model_path, sess_options=sess_options,
                                                  providers=["CPUExecutionProvider"])
            ...
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        # returns [{"class_id", "class_name", "confidence", "bbox": [x1, y1, x2, y2]}]
        ...
```

**ONNX postprocessing:** Output shape `[1, 84, 8400]` → transpose to `[8400, 84]`; columns `[:4]` are `[cx, cy, w, h]` (centre-format); columns `[4:]` are per-class logits. Vectorised argmax over class columns gives `class_id` and `confidence`. NMS via `cv2.dnn.NMSBoxes`.

**Preprocessing pipeline:** BGR → RGB → `float32 / 255.0` → NCHW → `[1, 3, 640, 640]`.

**Done when:** `YOLODetector("data/models/yolov8n.onnx").detect(frame)` returns a list of dicts with non-negative integer `bbox` coordinates that lie within the frame dimensions; no `ultralytics` import is triggered on the ONNX path.

---

## Block 3 — ByteTrack multi-object tracking

**Goal:** Maintain stable, persistent track identities across frames using IoU-based association and a Kalman filter. No ReID network — IoU only — to stay within CPU edge budget.

**File:** `src/tracking/byte_tracker.py`

**Key classes:**

`KalmanBoxTracker` — 8D state: `[cx, cy, aspect_ratio, h, vx, vy, va, vh]`. Transition matrix `F` is the identity with velocity-to-position coupling (`F[i, i+4] = 1.0` for `i` in `0..3`). Measurement matrix `H = eye(4, 8)`. Track IDs formatted as `T-{id:03d}`.

`ObjectTracker` (ByteTrack) — two-pass Hungarian matching using `scipy.optimize.linear_sum_assignment`:
1. **Pass 1:** high-confidence detections (`conf ≥ track_thresh`, default 0.5) matched against all active trackers.
2. **Pass 2:** low-confidence detections (`0.1 ≤ conf < track_thresh`) matched against unmatched trackers from Pass 1.

```python
def update(self, detections: List[Dict], frame_size: Tuple[int, int]) -> List[Dict]:
    # returns [{"track_id", "class_name", "confidence", "bbox": [x1,y1,x2,y2], "centroid": [cx,cy]}]
    ...
```

IoU matrix computation is fully vectorised with NumPy broadcasting (no Python loops over box pairs).

**Done when:** A track started at frame 0 retains the same `track_id` for 60 consecutive frames with no interruption when the same detection is present every frame; `KalmanBoxTracker.count` resets correctly between test runs.

---

## Block 4 — Zone configuration

**Goal:** Define restricted area polygons and their associated anomaly rules in a YAML file that operators can edit without touching source code.

**File:** `config/zones.yaml`

```yaml
zones:
  - name: R-01
    label: "Machinery Exclusion Zone"
    polygon: [[120, 250], [480, 250], [480, 600], [120, 600]]
    rules:
      intrusion: true
      loitering_threshold_seconds: 10
      crowd_threshold: 1
  - name: R-02
    label: "High-Voltage Equipment Area"
    polygon: [[310, 200], [640, 200], [640, 580], [310, 580]]
    rules:
      intrusion: true
      loitering_threshold_seconds: 30
      crowd_threshold: 2
  - name: GATE-01
    label: "Entrance Gate Monitoring"
    polygon: [[550, 300], [850, 300], [850, 720], [550, 720]]
    rules:
      intrusion: false
      loitering_threshold_seconds: 60
      crowd_threshold: 5
```

Supporting configs: `config/tracker.yaml` (ByteTrack parameters), `config/rules.yaml` (default rule thresholds).

**Done when:** `load_zones("config/zones.yaml")` returns a list of three zone dicts with all keys present; `ZoneMonitor(zones)` instantiates without error with these definitions.

---

## Block 5 — Zone monitor

**Goal:** For each video frame, determine which active tracks are inside each configured zone polygon using exact point-in-polygon geometry (not bounding-box overlap).

**File:** `src/anomaly/zone_monitor.py`

**Key functions:**
```python
def load_zones(config_path: str = "config/zones.yaml") -> List[Dict[str, Any]]:
    # reads YAML, returns raw zone list

class ZoneMonitor:
    def __init__(self, zones: List[Dict]):
        # builds shapely.geometry.Polygon for each zone with ≥ 3 vertices
        ...
    def check_tracks(self, tracks: List[Dict]) -> List[Dict]:
        # for each track centroid [cx, cy]: Point(cx, cy)
        # zone hit if polygon.contains(point) or polygon.touches(point)
        # returns [{"track_id", "zone_name", "zone_label", "rules", "bbox", "centroid", "confidence"}]
        ...
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        # semi-transparent red fill via cv2.addWeighted(alpha=0.25)
        # zone name label at top-left vertex with white shadow + coloured text
        ...
```

**Centroid vs bbox:** The check uses the track centroid (centre of the bounding box), not bounding-box overlap. This prevents false positives when a person standing outside a zone has their bounding box partially overlapping it.

**Done when:** A centroid placed at the known interior of a polygon returns a zone hit; a centroid placed 5px outside the same polygon returns no hit; `draw_zones` produces a frame with a visibly different region.

---

## Block 6 — Anomaly rules engine

**Goal:** Implement three stateful anomaly rules — INTRUSION, LOITERING, CROWD — with edge-triggering (each fires once per episode) and departure grace period to handle per-frame detection jitter.

**File:** `src/anomaly/rules_engine.py`

**Key class — `RulesEngine`:**
```python
class RulesEngine:
    def __init__(self, departure_grace_frames: int = 5, crowd_cooldown_seconds: float = 1.0):
        self._dwell_start: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._intrusion_fired: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self._loitering_fired: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self._tracks_last_seen: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._zone_occupancy: Dict[str, Set[str]] = defaultdict(set)
        self._last_crowd_alert_time: Dict[str, float] = {}
        ...
    def evaluate(self, zone_hits: List[Dict], frame_idx: int, fps: float) -> List[Dict]:
        ...
```

**Rule semantics:**
- **INTRUSION:** Fires once when a track first enters a zone (`intrusion_fired` flag; resets on departure after grace period).
- **LOITERING:** Fires once when `dwell_seconds ≥ loitering_threshold_seconds`; `dwell_start` persists across brief absences within the grace window.
- **CROWD:** Fires when `zone_occupancy > crowd_threshold`; suppressed within `crowd_cooldown_seconds` to prevent per-frame alert storms (1-second hysteresis).

**Departure detection:** At each frame, for each track previously in a zone: if `(frame_idx - last_seen) > departure_grace_frames`, purge all state for that track+zone pair.

**Done when:** See Block 11 test suite — all four test cases pass without modification.

---

## Block 7 — Alert generation & evidence capture

**Goal:** Produce a fully typed, persisted alert record with an evidence frame thumbnail every time the rules engine fires an event.

**File:** `src/alerts/alert_generator.py`

**Pydantic schema — `Alert`:**
```python
class Alert(BaseModel):
    alert_id: str          # "ALERT-0001"
    timestamp: str         # ISO-8601 UTC
    source: str            # video stream identifier
    frame_number: int
    track_id: str
    anomaly_type: Literal["INTRUSION", "LOITERING", "CROWD"]
    zone_name: str
    zone_label: str
    confidence: Optional[float] = 1.0
    bbox_at_trigger: List[int]   # [x1, y1, x2, y2]
    dwell_seconds: Optional[float] = None
    zone_count: Optional[int] = None
```

**`AlertGenerator`:**
- `_load_counter()`: scans `alerts/ALERT-*.json` on startup; parses the numeric suffix to resume numbering across restarts (idempotent — no alert overwrite on server restart).
- `generate(event, frame)`: increments counter, builds `Alert`, writes `ALERT-{n:04d}.json`, saves annotated evidence thumbnail to `alerts/frames/ALERT-{n:04d}.jpg` (red bounding box + header text drawn on copy of frame), pushes to in-memory `queue.Queue`.

**Done when:** `alert_gen.generate(event, frame)` writes a valid JSON file readable by `Alert.model_validate(json.load(f))`; counter correctly resumes from 5 when 5 prior files exist in the output directory.

---

## Block 8 — CLI pipeline integration

**Goal:** Wire all components into a single end-to-end callable function and expose it as a CLI entry point.

**File:** `src/main.py`

**Pipeline function `run_pipeline(...)`:**
```
StreamReader → YOLODetector → ObjectTracker → ZoneMonitor → RulesEngine → AlertGenerator
```

Execution loop per frame:
1. `detector.detect(frame)` → raw detections
2. `tracker.update(detections, reader.frame_size)` → active tracks
3. `monitor.check_tracks(tracks)` → zone hits
4. `rules.evaluate(zone_hits, frame_idx, reader.fps)` → triggered events
5. `alert_gen.generate(event, frame)` for each event
6. Optional: annotate frame with zone overlays, track boxes, centroid dots, HUD stats and write to video file

**Model path resolution order:** CLI `--model` arg → `MODEL_PATH` env var → `data/models/yolov8n.onnx` → `data/models/yolov8n.pt`.

**Done when:** `python src/main.py --source data/samples/factory_floor_sample.mp4 --max-frames 60` runs to completion, prints per-frame logs, exits with code 0, and generates at least one alert JSON if zones are configured.

---

## Block 9 — ONNX export & inference benchmark

**Goal:** Provide reproducible tooling to export the PyTorch weights to ONNX and benchmark inference latency head-to-head.

**Files:**
- `src/optimization/model_export.py`
- `src/optimization/benchmark.py`

**Export (`model_export.py`):**
```python
def export_to_onnx(pt_path: str, output_path: str, imgsz: int = 640):
    model = YOLO(pt_path)
    exported_file = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    shutil.copy(exported_file, output_path)
```
ONNX opset 12, `simplify=True` (onnx-simplifier fuses redundant ops); `dynamic=False` for static input shape at the edge device.

**Benchmark (`benchmark.py`):** 5-frame warmup before timing; `time.perf_counter()` per-frame wall-clock; 100-frame sample. Reports mean, max, and effective FPS for `.pt` and `.onnx` side-by-side.

**Target results:**
- PyTorch `.pt`: mean ~26–58ms/frame, high variance (up to 110ms spikes from Python/PyTorch warmup)
- ONNX Runtime `.onnx`: mean ~31ms/frame, tightly bounded max ~37ms/frame

The ONNX path trades a marginally higher mean for dramatically lower variance — the relevant metric for real-time 30 FPS edge processing where dropped frames matter more than average throughput.

**Done when:** `python src/optimization/benchmark.py` completes on both `.pt` and `.onnx` weights and prints a results table with FPS > 25 for ONNX; `python src/optimization/model_export.py` produces a valid `.onnx` file loadable by `onnxruntime.InferenceSession`.

---

## Block 10 — Streamlit dashboard

**Goal:** Interactive live surveillance console with four tabs: real-time annotated video feed, alert log with evidence frames, zone boundary inspector, and historical performance analytics.

**File:** `dashboard/app.py`

**Four tabs:**
1. **🎥 Live Monitor:** Runs the full pipeline in-process on a selected video; streams annotated frames to `st.empty()` image placeholder; live occupancy table per zone; inference time HUD (rolling 30-frame mean).
2. **📋 Alert Log & Evidence:** Loads all `ALERT-*.json` files from `alerts/`; displays as a filtered dataframe; evidence frame thumbnail viewer per selected alert with raw JSON inspection.
3. **📍 Zone Boundaries:** Reads `config/zones.yaml` and displays each zone's polygon vertices and rule parameters in expandable panels.
4. **📊 Performance & Analytics:** Bar charts of incident frequency by anomaly type and by zone (rendered with `st.bar_chart` from a pandas `value_counts()` series).

**Sidebar controls:** video source selector (sample clip / custom path / webcam device), model selector (ONNX or .pt), confidence and NMS IoU sliders, max-frames slider, start button.

**Done when:** `streamlit run dashboard/app.py` starts without import error; all four tabs render without exception; clicking "▶ Start Real-Time Analysis" processes frames and updates the video placeholder in real time.

---

## Block 11 — Test suite

**Goal:** Cover the three core algorithmic components with deterministic unit tests that use synthetic inputs only (no real video dependency).

**Files:**
- `tests/test_rules_engine.py` — four tests covering INTRUSION edge-trigger (fires once, not on frame 2), LOITERING edge-trigger (fires exactly at threshold, not before or after), jitter tolerance (dwell timer survives 3-frame absence within grace window), CROWD rule (not triggered at threshold, triggered above it, suppressed within cooldown).
- `tests/test_zone_monitor.py` — point-in-polygon tests with known geometric inputs: interior point returns hit, exterior point returns no hit, boundary-touching point returns hit.
- `tests/test_alert_schema.py` — Pydantic validation tests: a fully populated `Alert.model_validate(data)` round-trips through `model_dump()` without error; an invalid `anomaly_type` raises `ValidationError`.

**Key test pattern (rules engine):**
```python
def test_loitering_rule_fires_exactly_once_at_threshold():
    engine = RulesEngine(departure_grace_frames=5)
    fps = 30.0
    threshold_seconds = 10.0  # 300 frames
    hit = {"track_id": "T-002", "zone_name": "R-02", ...,
           "rules": {"intrusion": False, "loitering_threshold_seconds": threshold_seconds}}
    for f in range(300):
        evs = engine.evaluate([hit], frame_idx=f, fps=fps)
        assert not any(e["anomaly_type"] == "LOITERING" for e in evs)
    evs_300 = engine.evaluate([hit], frame_idx=300, fps=fps)
    loitering = [e for e in evs_300 if e["anomaly_type"] == "LOITERING"]
    assert len(loitering) == 1
    assert loitering[0]["dwell_seconds"] >= 10.0
```

**Done when:** `pytest` reports all tests passing on a clean environment with no real video files or model weights required.

---

## Block 12 — CI pipeline & documentation

**Goal:** Automate test execution on every push and pull request; publish reference documentation for deployment and calibration.

**File:** `.github/workflows/tests.yml`

```yaml
name: AEGIS Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pytest
```

**Documentation files:**
- `docs/architecture.md` — component interaction diagram and data flow description
- `docs/detection-methodology.md` — YOLOv8n model selection rationale, ONNX postprocessing pipeline, NMS parameters
- `docs/edge-optimization.md` — INT8 quantization path (9ms/frame, ~3% mAP drop), ONNX Runtime execution providers, Jetson Nano deployment notes
- `docs/scene-notes.md` — camera angle assumptions, zone calibration procedure, known limitations (occlusion, lighting variance)
- `docs/setup.md` — quick-start installation, zone calibration steps, CLI and dashboard launch commands

**Done when:** GitHub Actions matrix passes on both Python versions; all five documentation files exist and link correctly from `README.md`.
