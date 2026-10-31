import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yaml

from src.ingestion.stream_reader import StreamReader
from src.detection.yolo_detector import YOLODetector
from src.tracking.byte_tracker import ObjectTracker
from src.anomaly.zone_monitor import ZoneMonitor, load_zones
from src.anomaly.rules_engine import RulesEngine
from src.alerts.alert_generator import AlertGenerator


st.set_page_config(
    page_title="AEGIS | Edge Surveillance Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #E2E8F0;
    }
    .alert-badge-intrusion {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .alert-badge-loitering {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .alert-badge-crowd {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_all_alerts(alert_dir: str = "alerts/"):
    """Reads all saved alert JSON records from the alert directory."""
    path = Path(alert_dir)
    if not path.exists():
        return []
    alert_files = sorted(path.glob("ALERT-*.json"), key=os.path.getmtime, reverse=True)
    alerts = []
    for f in alert_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                alerts.append(json.load(fp))
        except Exception:
            continue
    return alerts


# Sidebar Configuration
st.sidebar.title("🛡️ AEGIS Config")
st.sidebar.markdown("**Adaptive Edge-Guided Intelligence for Surveillance**")

# Samples detection
sample_files = list(Path("data/samples").glob("*.mp4"))
sample_options = [str(p) for p in sample_files]
if not sample_options:
    sample_options = ["data/samples/factory_floor_sample.mp4"]

source_type = st.sidebar.radio("Input Stream Type", ["Sample Video", "Custom Path / RTSP", "Webcam Device"])
if source_type == "Sample Video":
    video_source = st.sidebar.selectbox("Select Sample Clip", sample_options)
elif source_type == "Custom Path / RTSP":
    video_source = st.sidebar.text_input("Stream URL or File Path", "data/samples/factory_floor_sample.mp4")
else:
    cam_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
    video_source = str(cam_idx)

# Model selection
models_available = []
if Path("data/models/yolov8n.onnx").exists():
    models_available.append("data/models/yolov8n.onnx (ONNX Runtime - Edge Optimized)")
if Path("data/models/yolov8n.pt").exists():
    models_available.append("data/models/yolov8n.pt (PyTorch Native)")
if not models_available:
    models_available = ["data/models/yolov8n.pt"]

selected_model_label = st.sidebar.selectbox("Inference Engine", models_available)
model_path = selected_model_label.split(" ")[0]

conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
max_frames_to_run = st.sidebar.slider("Max Frames to Process (Re-analysis)", min_value=50, max_value=600, value=180, step=30)

start_analysis = st.sidebar.button("▶ Start Real-Time Analysis", type="primary")

# Main Header
st.markdown('<p class="main-title">🛡️ AEGIS Live Surveillance Operations Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time edge analytics: YOLOv8n + ByteTrack + ROI Polygon Engine</p>', unsafe_allow_html=True)

# Top KPI metrics
existing_alerts = load_all_alerts()
total_alerts_count = len(existing_alerts)
intrusion_count = sum(1 for a in existing_alerts if a.get("anomaly_type") == "INTRUSION")
loitering_count = sum(1 for a in existing_alerts if a.get("anomaly_type") == "LOITERING")
crowd_count = sum(1 for a in existing_alerts if a.get("anomaly_type") == "CROWD")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Alerts", total_alerts_count)
with m2:
    st.metric("Zone Intrusions", intrusion_count)
with m3:
    st.metric("Loitering Violations", loitering_count)
with m4:
    st.metric("Crowd Anomalies", crowd_count)
with m5:
    st.metric("Edge Latency (Target)", "≤ 20 ms")

tab_live, tab_alerts, tab_zones, tab_analytics = st.tabs(["🎥 Live Monitor", "📋 Alert Log & Evidence", "📍 Zone Boundaries", "📊 Performance & Analytics"])

with tab_live:
    col_video, col_status = st.columns([2, 1])

    with col_status:
        st.subheader("Active Scene Status")
        status_box = st.empty()
        occupancy_table = st.empty()

    with col_video:
        video_placeholder = st.empty()
        progress_bar = st.progress(0)

    if start_analysis:
        try:
            reader = StreamReader(video_source)
            detector = YOLODetector(model_path, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
            tracker_cfg = {"track_thresh": conf_thresh, "track_buffer": 30, "match_thresh": 0.8}
            tracker = ObjectTracker(tracker_cfg)
            zones = load_zones("config/zones.yaml")
            monitor = ZoneMonitor(zones)
            rules = RulesEngine()
            alert_gen = AlertGenerator(output_dir="alerts/", source=str(video_source))

            frame_times = []
            for frame_idx, frame in reader.frames():
                if frame_idx >= max_frames_to_run:
                    break

                t0 = time.perf_counter()
                detections = detector.detect(frame)
                tracks = tracker.update(detections, reader.frame_size)
                zone_hits = monitor.check_tracks(tracks)
                events = rules.evaluate(zone_hits, frame_idx, reader.fps)

                for ev in events:
                    alert_gen.generate(ev, frame=frame)

                dt = (time.perf_counter() - t0) * 1000.0
                frame_times.append(dt)

                # Annotate frame
                annotated = monitor.draw_zones(frame.copy())
                for t in tracks:
                    x1, y1, x2, y2 = t["bbox"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{t['track_id']}",
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cx, cy = t["centroid"]
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                progress_bar.progress((frame_idx + 1) / max_frames_to_run)

                # Update live stats table
                zone_counts = {}
                for z in zones:
                    name = z["name"]
                    c = sum(1 for h in zone_hits if h["zone_name"] == name)
                    zone_counts[name] = {
                        "Zone": name,
                        "Label": z.get("label", ""),
                        "Current Occupancy": c,
                        "Crowd Limit": z.get("rules", {}).get("crowd_threshold", "-"),
                        "Intrusion Rule": "Active" if z.get("rules", {}).get("intrusion") else "Off",
                    }
                occupancy_table.dataframe(pd.DataFrame(list(zone_counts.values())), hide_index=True)

                avg_lat = np.mean(frame_times[-30:]) if frame_times else dt
                status_box.markdown(f"""
                - **Frame:** {frame_idx} / {max_frames_to_run}
                - **Active Tracks:** {len(tracks)}
                - **Inference Time:** `{avg_lat:.1f} ms` (FPS: `{1000.0 / max(avg_lat, 1.0):.1f}`)
                - **Hardware:** CPU Execution Provider (Edge)
                """)

            reader.release()
            st.success(f"Processing complete! Processed {max_frames_to_run} frames. Mean latency: {np.mean(frame_times):.1f} ms/frame.")
            st.rerun()

        except Exception as e:
            st.error(f"Error during video processing: {e}")
    else:
        video_placeholder.info("Click '▶ Start Real-Time Analysis' in the sidebar to run the edge pipeline.")

with tab_alerts:
    st.subheader("Real-Time Incident Records")
    current_alerts = load_all_alerts()
    if not current_alerts:
        st.info("No alert records logged yet. Run the pipeline to capture events.")
    else:
        df_alerts = pd.DataFrame(current_alerts)
        st.dataframe(
            df_alerts[["alert_id", "timestamp", "anomaly_type", "zone_name", "zone_label", "track_id", "dwell_seconds", "zone_count"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Evidence Frame Inspection")
        selected_alert_id = st.selectbox("Select Alert to Inspect", [a["alert_id"] for a in current_alerts])
        if selected_alert_id:
            thumb_path = Path("alerts/frames") / f"{selected_alert_id}.jpg"
            selected_alert_data = next((a for a in current_alerts if a["alert_id"] == selected_alert_id), None)
            
            c_img, c_meta = st.columns([2, 1])
            with c_img:
                if thumb_path.exists():
                    st.image(str(thumb_path), caption=f"Evidence Snapshot for {selected_alert_id}", use_container_width=True)
                else:
                    st.warning("Frame thumbnail not found for this alert.")
            with c_meta:
                if selected_alert_data:
                    st.json(selected_alert_data)

with tab_zones:
    st.subheader("Configured ROI Geometries (`config/zones.yaml`)")
    zones_list = load_zones("config/zones.yaml")
    for z in zones_list:
        with st.expander(f"📍 {z['name']} — {z.get('label', '')}", expanded=True):
            st.write(f"**Polygon Vertices:** `{z.get('polygon')}`")
            st.write(f"**Associated Security Rules:** `{z.get('rules')}`")

with tab_analytics:
    st.subheader("Historical Analytics & Benchmarks")
    if current_alerts:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Incident Frequency by Type**")
            type_counts = pd.Series([a["anomaly_type"] for a in current_alerts]).value_counts()
            st.bar_chart(type_counts)
        with c2:
            st.markdown("**Incident Frequency by Zone**")
            zone_counts = pd.Series([a["zone_name"] for a in current_alerts]).value_counts()
            st.bar_chart(zone_counts)
    else:
        st.info("Run surveillance feeds to populate incident distribution analytics.")
