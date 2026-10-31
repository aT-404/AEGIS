# Detection & Tracking Methodology

## Detection Model Selection: YOLOv8n

For real-time surveillance on edge hardware (laptops, mini-PCs, Intel NUCs, Raspberry Pi 4, Jetson Nano), model weight and compute latency dictate feasibility.

| Model Variant | Parameters | FLOPs (640px) | CPU Latency (PyTorch) | CPU Latency (ONNX Runtime) |
|---|---|---|---|---|
| **YOLOv8n (Nano)** | **3.2M** | **8.7 G** | **~58 ms** | **~14 ms** |
| YOLOv8s (Small) | 11.2M | 28.6 G | ~120 ms | ~38 ms |
| YOLOv8m (Medium) | 25.9M | 78.9 G | ~260 ms | ~85 ms |

YOLOv8n was selected because it achieves the critical sub-20ms threshold required for real-time 30 FPS processing on standard multi-core CPUs. The nano model achieves ~37.3 mAP on COCO, which is more than sufficient for detecting pedestrians, workers, and machinery in typical industrial scenes.

### Confidence and NMS Tuning

- **Confidence Threshold (`conf`):** Configured to `0.40`. Lower thresholds (<0.3) introduce false positives on textures and equipment shadows; higher thresholds (>0.6) risk dropping partially occluded personnel.
- **IoU / NMS Threshold (`iou`):** Configured to `0.50` for Non-Maximum Suppression to balance overlapping person detections in moderate crowd conditions.

## Tracking: ByteTrack vs. DeepSORT

AEGIS employs **ByteTrack** for identity persistence across consecutive frames.

| Dimension | ByteTrack | DeepSORT |
|---|---|---|
| Association Mechanism | Dual-threshold IoU + Kalman Filter | Visual ReID Features + Kalman Filter |
| Compute Overhead | ~1–2 ms (pure CPU linear algebra) | 15–30 ms (secondary CNN feature extraction) |
| Low-Score Retention | Yes (second-stage match recovers occluded boxes) | No (drops low-confidence detections immediately) |
| Edge Viability | High (negligible CPU footprint) | Low (ReID bottleneck on CPU) |

In industrial factory floors and access gates, background textures remain relatively static and camera angles are fixed overhead or corner-mounted. ByteTrack's motion-based association with low-score detection recovery delivers low ID switch rates without the heavy ReID compute penalty.
