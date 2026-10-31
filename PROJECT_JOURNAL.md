# Project Journal — AEGIS (Adaptive Edge-Guided Intelligence for Surveillance)

A running engineering log documenting the architectural rationale, edge constraints, testing breakthroughs, and system design decisions made during the development of AEGIS.

---

### Entry 1 — The idea

My previous work at SRISHTI Robotics involved pose estimation for a robotic arm — estimating joint angles from a camera feed so the arm could be positioned correctly. That was the first time I seriously worked with a real-time vision pipeline, and the thing I took away from it was less about the specific model and more about the architecture: detecting something in a frame is only half the problem. Knowing whether the thing that appeared in frame 1200 is the same thing that appeared in frame 847 is the harder problem.

I've been thinking about what comes next in industrial computer vision. The obvious next level is surveillance — not just arm joints but people, equipment, and zones.

L&T runs factories, construction sites, and power plants. Every one of those environments has restricted zones: high-voltage areas, machinery exclusion zones, loading docks with forklift traffic. They all have CCTV cameras. The operational problem is not hardware — it's analysis. Nobody is watching 200 camera feeds at once. Incidents are discovered in hindsight.

The commercial answer to this is cloud video analytics: pipe the stream to a cloud service, get alerts back. The technical problem with that is latency (a 2-second cloud round-trip is too long for a safety event), bandwidth cost (HD video streams from 200 cameras is not cheap), and data sensitivity (factory floor footage is proprietary). Edge deployment — running the inference on the device, sending only alerts upstream — solves all three. That's what I set out to build.

### Entry 2 — Research

The obvious starting point for the detection model was YOLO. YOLOv8 is the current standard — easy to work with via Ultralytics, good performance across model sizes, and critically, the nano variant (YOLOv8n) is specifically designed for resource-constrained deployment. 3.2M parameters, ~8ms per frame on a decent CPU with ONNX Runtime. That's fast enough to process 30fps video in real-time on a laptop, which means it's fast enough on an edge device with similar-class hardware.

For tracking, I started with DeepSORT (the standard reference for multi-object tracking) and then switched to ByteTrack after reading the ByteTrack paper. The key difference: DeepSORT uses a re-identification (ReID) feature extractor to match detections to tracks — which means running a second neural network per frame. ByteTrack uses only bounding box IoU and a Kalman filter for motion prediction, with no ReID network. Lower accuracy on very crowded scenes, but 40–60% faster. For an edge deployment on a factory floor (not a crowded city street), that tradeoff makes ByteTrack the right choice.

The anomaly logic is the part I had to design from scratch. There's no standard library for "person has been in this polygon for 30 seconds" — that's application-level logic that depends entirely on what counts as anomalous in the target environment. I settled on three rules that cover the most common industrial surveillance requirements: zone intrusion (something entered a restricted area), loitering (something stayed too long), and crowd formation (too many things at once). Every other anomaly type I could think of is a variant of one of these three.

### Entry 3 — Why edge, and what that actually means

"Edge deployment" is sometimes used loosely to mean "I ran it locally without a cloud API." I want to be precise about what I mean: the ONNX export step is real, the ONNX Runtime inference benchmark is real, and the performance targets are achievable on a Jetson Nano, Raspberry Pi 4 B+, or Intel IPC with the right optimizations (INT8 quantization, reduced input resolution).

I'm not deploying to an actual Jetson Nano hardware unit in this phase — I don't have one on my desk. But I built the pipeline in a way where that deployment step requires only a runtime swap (ONNX Runtime with the CUDA or TensorRT execution provider instead of the CPU provider), not an architecture rewrite. The benchmark results document the actual measured inference times on local CPU, not estimated figures.

### Entry 4 — Developing the Plan & System Review (with AI)

Before committing code, I drafted the initial system architecture and used AI as a sounding board to stress-test the implementation plan, uncover edge cases, and challenge architectural assumptions before writing the rules engine. Key improvements that came out of this planning phase:

- **Loitering Edge-Triggering:** My initial loitering concept fired on every single frame a track was inside a zone past the threshold. That meant one person loitering for 5 minutes would generate 150 alert JSON files (at 30fps, after the 30-second threshold). We caught this during planning and resolved it by implementing an edge-trigger: the rule fires exactly once when dwell crosses the threshold, and resets only if the track leaves the zone and re-enters.
- **Compute Overhead Tradeoff:** I originally planned to use DeepSORT. Walking through the CPU compute overhead of the ReID feature extractor confirmed it would bottleneck edge throughput. Switching to ByteTrack ensured the pipeline stays well within 30 FPS CPU limits.
- **Config Decoupling:** My initial draft had zone definitions hardcoded as pixel coordinate lists in Python scripts. Moving them to `config/zones.yaml` makes the system truly field-deployable without modifying source code.
- **Evidence Capture:** An alert JSON without visual evidence is difficult to verify in an industrial operations center. Added automated frame thumbnail snapshots (`alerts/frames/ALERT-XXXX.jpg`) at the moment of trigger.
- **Stateful Counter Idempotency:** Added `_load_counter()` to initialize alert numbering from existing files on disk, ensuring server restarts never overwrite previous incident logs.

### Entry 5 — The sliding window vs fixed bin problem

The most critical engineering challenge in the rules engine was the loitering dwell timer. My first implementation computed dwell time as `current_frame / fps - entry_frame / fps`. This works in theory, but has a hidden flaw: if a track briefly exits the polygon (one frame where the centroid crosses the boundary due to detection jitter) and immediately re-enters, the dwell timer resets to zero. A person standing motionless at the zone boundary could oscillate across it and never trigger loitering.

Fixed by adding a grace period: tracks that exit a zone and re-enter within 5 frames keep their original dwell start time. The departure check now happens at the end of each frame processing cycle, not immediately on exit. This small change makes the loitering rule robust against the detection jitter that happens in any real video stream — bounding boxes shift slightly frame-to-frame even when the person isn't moving.

Same class of problem appeared in the crowd rule: a group of people near the threshold count (say, threshold is 3 and 4 people are present) was generating a crowd alert every frame, because the zone occupancy oscillated between 3 and 4 due to one person partially crossing the zone boundary. Fixed with a 1-second hysteresis: the crowd rule doesn't trigger again within 1 second of the last crowd trigger for the same zone.

These weren't theoretical bugs. They surfaced immediately when I ran the pipeline on the entrance gate sample video and saw 340 crowd alerts generated for a 90-second clip. Real anomaly detection systems need to produce actionable alerts, not noise.

### Entry 6 — MOT17 tracker validation

Running the tracker on MOT17-04 and measuring MOTA/IDF1 scores was added to establish quantitative tracking benchmarks. MOTA and IDF1 are standard metrics for multi-object tracking research:

The scores (MOTA ≈ 53, IDF1 ≈ 58) are solid for a CPU-bound setup. A ByteTrack implementation with a dedicated pedestrian-trained detector reaches ~77 MOTA. Here, I'm using a general-purpose YOLOv8n model running on standard CPU without GPU acceleration. The point is that the system produces measurable tracking quality with verifiable data rather than qualitative claims.

### Entry 7 — ONNX benchmark

Running the `.pt` (Ultralytics) and `.onnx` (ONNX Runtime) versions side by side on 100 frames:

- Ultralytics `.pt`: mean ~26–58ms/frame (Python overhead, PyTorch warmup, high variance spikes up to 110ms).
- ONNX Runtime `.onnx`: mean ~31ms/frame, tightly bounded max ~37ms/frame.

The stability improvement confirms that ONNX export is a meaningful operational step — it provides steady, deterministic frame times crucial for real-time 30 FPS processing on CPU hardware without dropped frames.

I also tested INT8 quantization using ONNX Runtime's quantization API. Results: 9ms/frame average, but about a 3% drop in mAP on a small validation set. For a zone intrusion application where a missed detection is more costly than a slightly degraded bounding box, I kept FP32 precision for default deployment and documented the quantization path in `docs/edge-optimization.md`.

### Entry 8 — System Status

Pipeline works end-to-end on all four sample clips. All three anomaly rules fire correctly and produce non-redundant alerts. The ONNX Runtime inference runs at real-time speeds on the development machine. The Pydantic alert schema validates all generated alerts without errors.

The MOT17 tracker validation gives the project a quantitative baseline, and the benchmarking tools give it an edge-deployment profile grounded in measured numbers.

### Entry 9 — Testing and CI

The test suite has three files. The zone monitor tests were straightforward — I could compute the expected point-in-polygon results analytically. The rules engine tests required building a frame-by-frame simulation: I generate a list of synthetic zone-hit events (track T-001 entering zone R-02 at frame 0, staying for 35 simulated seconds at 30fps = 1050 frames) and assert the loitering event fires once at frame 900 (30 seconds × 30fps). Simulating the sequence was more work than the actual rule — but it makes the test deterministic and removes any dependency on real video.

One genuine catch from the test: the loitering trigger was off by one frame. The rule was comparing `dwell >= threshold` rather than `dwell - (1/fps) < threshold <= dwell`, which meant the single-trigger logic was checking against the wrong edge. The unit test caught it immediately.

---

*The core takeaway from building AEGIS: the most challenging engineering is not just running YOLO inference. The real work is everything that happens after the detector returns a bounding box — maintaining identity across occlusions, checking zone membership efficiently without per-pixel operations, and generating alerts that are actionable rather than redundant.*
