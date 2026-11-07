import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from src.ingestion.stream_reader import StreamReader
from src.detection.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ObjectTracker
from src.anomaly.zone_monitor import ZoneMonitor, load_zones
from src.anomaly.rules_engine import RulesEngine
from src.alerts.alert_generator import AlertGenerator


def run_pipeline(
    source: str = "data/samples/factory_floor_sample.mp4",
    output_dir: str = "alerts/",
    write_video: bool = False,
    video_output_path: str = "outputs/annotated_output.mp4",
    model_path: str = None,
    max_frames: int = None,
):
    """Executes the end-to-end AEGIS edge surveillance pipeline."""
    if model_path is None:
        env_model = os.getenv("MODEL_PATH")
        if env_model and Path(env_model).exists():
            model_path = env_model
        elif Path("data/models/yolov8n.onnx").exists():
            model_path = "data/models/yolov8n.onnx"
        else:
            model_path = "data/models/yolov8n.pt"

    print(f"==================================================")
    print(f"AEGIS - Adaptive Edge-Guided Surveillance System")
    print(f"Video Source : {source}")
    print(f"Model Path   : {model_path}")
    print(f"Alert Output : {output_dir}")
    print(f"==================================================")

    # Initialize components
    reader = StreamReader(source)
    detector = YOLODetector(model_path)

    tracker_cfg = {}
    if Path("config/tracker.yaml").exists():
        with open("config/tracker.yaml", "r", encoding="utf-8") as f:
            tracker_cfg = yaml.safe_load(f) or {}
    tracker = ObjectTracker(tracker_cfg)

    rules_cfg = {}
    if Path("config/rules.yaml").exists():
        with open("config/rules.yaml", "r", encoding="utf-8") as f:
            rules_cfg = yaml.safe_load(f) or {}

    zones = load_zones("config/zones.yaml")
    monitor = ZoneMonitor(zones)
    rules = RulesEngine(rules_cfg=rules_cfg)
    alert_gen = AlertGenerator(output_dir=output_dir, source=str(source))

    writer = None
    if write_video:
        out_file = Path(video_output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, reader.fps, reader.frame_size)

    total_alerts = 0
    frame_count = 0

    try:
        for frame_idx, frame in reader.frames():
            if max_frames and frame_idx >= max_frames:
                break
            frame_count += 1

            # 1. Detection
            detections = detector.detect(frame)

            # 2. Multi-Object Tracking
            tracks = tracker.update(detections, reader.frame_size)

            # 3. Zone Boundary Containment
            zone_hits = monitor.check_tracks(tracks)

            # 4. Anomaly Rules Evaluation
            events = rules.evaluate(zone_hits, frame_idx, reader.fps)

            # 5. Alert Generation & Evidence Capture
            for event in events:
                alert = alert_gen.generate(event, frame=frame)
                total_alerts += 1
                print(
                    f"[ALERT #{total_alerts:04d}] {alert.alert_id}: {alert.anomaly_type} "
                    f"-> Track {alert.track_id} in {alert.zone_name} ({alert.zone_label}) "
                    f"at Frame {alert.frame_number}"
                )

            # 6. Optional Video Annotation
            if write_video and writer:
                annotated = monitor.draw_zones(frame.copy())

                # Draw track bounding boxes & IDs
                for track in tracks:
                    x1, y1, x2, y2 = track["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{track['track_id']} ({track.get('class_name', 'person')})"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cx, cy = track["centroid"]
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

                # Draw HUD stats
                hud_text = f"Frame: {frame_idx} | Tracks: {len(tracks)} | Alerts: {total_alerts}"
                cv2.putText(
                    annotated,
                    hud_text,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                writer.write(annotated)

            if frame_idx % 60 == 0:
                print(f"Processed frame {frame_idx} | Active tracks: {len(tracks)} | Alerts: {total_alerts}")

    finally:
        reader.release()
        if writer:
            writer.release()
            print(f"Annotated video written to {video_output_path}")

    print(f"Execution finished. Total frames processed: {frame_count}. Total alerts generated: {total_alerts}.")
    return total_alerts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS Edge Surveillance Engine")
    parser.add_argument("--source", default="data/samples/factory_floor_sample.mp4", help="Path to video or RTSP stream")
    parser.add_argument("--output", default="alerts/", help="Directory to save JSON alerts and thumbnails")
    parser.add_argument("--model", default=None, help="Path to YOLO weights (.onnx or .pt)")
    parser.add_argument("--write-video", action="store_true", help="Record annotated output video")
    parser.add_argument("--video-out", default="outputs/annotated_output.mp4", help="Path for output video")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames to process")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output_dir=args.output,
        write_video=args.write_video,
        video_output_path=args.video_out,
        model_path=args.model,
        max_frames=args.max_frames,
    )
