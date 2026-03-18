'''import os
import time
import streamlit as st

from backend.traffic_engine import run_traffic_system


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Based Traffic Signal Control",
    layout="wide"
)

st.title("🚦 AI-Based Traffic Signal Control")

st.markdown(
    "Upload lane videos. The system will process frames and stream output live."
)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload lane videos",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True
)

# -------------------------------------------------
# RUN BUTTON
# -------------------------------------------------
if st.button("Run System"):

    if not uploaded_files:
        st.error("Please upload at least one video.")
        st.stop()

    # ---------------- DIRECTORIES ----------------
    os.makedirs("videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ---------------- SAVE FILES ----------------
    lane_paths = []
    for i, file in enumerate(uploaded_files):
        path = f"videos/lane{i+1}.mp4"
        with open(path, "wb") as f:
            f.write(file.read())
        lane_paths.append(path)

    # ---------------- UI ELEMENTS ----------------
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    live_frame = st.empty()   # 🔴 LIVE STREAM CONTAINER

    MAX_FRAMES = 300

    # ---------------- CALLBACK ----------------
    def update_progress(frame_idx, frame=None):
        progress_bar.progress(min(frame_idx / MAX_FRAMES, 1.0))
        status_text.text(f"Processing frame {frame_idx}/{MAX_FRAMES}")

        if frame is not None:
            live_frame.image(
                frame,
                channels="BGR",
                use_column_width=True
            )

    # ---------------- RUN BACKEND ----------------
    with st.spinner("Processing video stream..."):
        output_path = run_traffic_system(
            lane_paths=lane_paths,
            yolo_weight="models/yolo11s.pt",
            output_path="outputs/output.avi",
            progress_callback=update_progress
        )

    time.sleep(1)

    # ---------------- FINAL OUTPUT ----------------
    if os.path.exists(output_path):
        st.success("✅ Processing completed successfully!")

        st.download_button(
            label="⬇ Download Output Video",
            data=open(output_path, "rb"),
            file_name="traffic_output.avi",
            mime="video/avi"
        )
    else:
        st.error("❌ Output video not found.")
'''

import os
import streamlit as st
from backend.traffic_engine import run_traffic_system

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Dashboard",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙️ Configuration")

uploaded_files = st.sidebar.file_uploader(
    "Upload lane videos",
    type=["mp4", "mpeg4"],
    accept_multiple_files=True
)

model_path = st.sidebar.text_input(
    "YOLOv11 weights path",
    value="Model/best.pt"
)

start_btn = st.sidebar.button("▶ Start System")
stop_btn = st.sidebar.button("⏹ Stop System")

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.markdown("## Yolo Based ATS Control System With Real Time Violation Monitoring")

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
_, center_col, right_col = st.columns([0.2, 6.5, 3.3])

with center_col:
    st.markdown("### 📹 Live feed")
    live_frame = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()

with right_col:
    st.markdown("### 🎛 Controls & status")
    system_state = st.empty()
    st.markdown("---")

    st.markdown("**Active green**")
    active_green_box = st.empty()

    st.markdown("**Remaining (s)**")
    remaining_time_box = st.empty()

    st.markdown("**Lane counts**")
    lane_count_box = st.empty()

    st.markdown("**Violations / Emergencies**")
    violation_box = st.empty()

# -------------------------------------------------
# RUN SYSTEM
# -------------------------------------------------
MAX_FRAMES = 3

if start_btn:

    if not uploaded_files:
        st.sidebar.error("Please upload at least one video.")
        st.stop()

    os.makedirs("videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    lane_paths = []
    for i, f in enumerate(uploaded_files):
        path = f"videos/lane{i+1}.mp4"
        with open(path, "wb") as out:
            out.write(f.read())
        lane_paths.append(path)

    system_state.success("🟢 Controller running")

    def update_progress(frame_idx, frame=None, stats=None):
        progress_bar.progress(min(frame_idx / MAX_FRAMES, 1.0))
        status_text.text(f"Processing frame {frame_idx}/{MAX_FRAMES}")

        if frame is not None:
            live_frame.image(frame, channels="BGR", width=900)

        if stats:
            active_green_box.markdown(f"### {stats['active_green']}")
            remaining_time_box.markdown(f"### {stats['remaining_time']}")
            lane_count_box.json(stats["lane_counts"])
            violation_box.markdown(
                f"🚨 **Violations:** {stats['violations']} &nbsp;&nbsp; "
                f"🚑 **Emergencies:** {stats['emergencies']}",
                unsafe_allow_html=True
            )

    output_path = run_traffic_system(
        lane_paths=lane_paths,
        yolo_weight=model_path,
        output_path="outputs/output.avi",
        progress_callback=update_progress
    )

    system_state.success("🟢 Controller stopped")

    if os.path.exists(output_path):
        st.download_button(
            "⬇ Download Output Video",
            data=open(output_path, "rb"),
            file_name="traffic_output.avi"
        )

if stop_btn:
    system_state.warning("🛑 Controller stopped by user")

'''from ultralytics import YOLO
model = YOLO("Model/best.pt")
print(model.names)'''

