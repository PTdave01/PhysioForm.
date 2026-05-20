import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="PhysioForm")
st.title("PhysioForm: Clinical Movement Tracker")

# --- 2. LOAD AI ONCE (Prevents crashing) ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- 3. MATH ENGINE ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: 
        angle = 360 - angle
    return angle

# --- 4. USER INTERFACE ---
exercise = st.selectbox("Select Exercise to Track", ["Squat", "Bicep Curl"])

# --- 5. VIDEO PROCESSOR (The Thread-Safe Engine) ---
class PhysioProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.exercise_type = "Squat" 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        
        for r in results:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            if len(keypoints) >= 16:
                # Dynamic Exercise Selection Logic
                if self.exercise_type == "Squat":
                    # Hip(11), Knee(13), Ankle(15)
                    pt1, pt2, pt3 = keypoints[11], keypoints[13], keypoints[15]
                    down_thresh, up_thresh = 160, 90
                else: # Bicep Curl
                    # Shoulder(5), Elbow(7), Wrist(9)
                    pt1, pt2, pt3 = keypoints[5], keypoints[7], keypoints[9]
                    down_thresh, up_thresh = 150, 40
                
                # Check if points exist on screen to avoid math errors
                if np.all(pt1) and np.all(pt2) and np.all(pt3):
                    angle = calculate_angle(pt1, pt2, pt3)
                    
                    # State Tracking
                    if angle > down_thresh:
                        self.stage = "down"
                    if angle < up_thresh and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        
                    # Draw UI directly on the video frame
                    cv2.putText(img, f"{self.exercise_type.upper()} REPS: {self.counter}", 
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(img, f"ANGLE: {int(angle)}", 
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. WEBRTC LAUNCHER (With Firewall Bypass) ---
ctx = webrtc_streamer(
    key="physioform",
    video_processor_factory=PhysioProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

# Pass the dropdown choice into the live video thread dynamically
if ctx.video_processor:
    ctx.video_processor.exercise_type = exercise
