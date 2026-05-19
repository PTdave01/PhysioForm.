import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. INITIALIZE THE MODEL ---
# This loads once when the app starts
model = YOLO('yolov8n-pose.pt')

# --- 2. TRIGONOMETRY ENGINE ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 3. DASHBOARD UI ---
st.title("PhysioForm: Clinical Movement Tracker")
st.write("Live Telehealth Dashboard")

# State variables for tracking
counter = 0
stage = None

def video_frame_callback(frame):
    global counter, stage
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLO detection
    results = model(img)
    
    for r in results:
        keypoints = r.keypoints.xy[0].cpu().numpy()
        if len(keypoints) >= 16:
            # SQUAT MATH: Hip(11), Knee(13), Ankle(15)
            hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
            angle = calculate_angle(hip, knee, ankle)
            
            # State tracking
            if angle > 160: 
                stage = "down"
            if angle < 90 and stage == "down":
                stage = "up"
                counter += 1
            
            # Draw feedback on video
            cv2.putText(img, f"REPS: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, f"ANGLE: {int(angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. START THE CAMERA ---
webrtc_streamer(
    key="physioform-camera",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
