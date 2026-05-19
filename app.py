import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

# --- DASHBOARD UI ---
st.title("PhysioForm: Clinical Movement Tracker")
st.write("Welcome to the live telehealth dashboard. Please select your exercise and grant camera permissions.")

# Dropdown menu for the user
exercise = st.selectbox("Select Exercise to Track", ["Bicep Curl", "Squat"])

# --- CAMERA VIDEO LOOP ---
# Import your math engine inside the function so the web server can use it
from engine import calculate_angle, model 
import numpy as np

counter = 0
stage = None

def video_frame_callback(frame):
    global counter, stage
    img = frame.to_ndarray(format="bgr24")
    
    # 1. Run YOLO detection
    results = model(img)
    
    for r in results:
        keypoints = r.keypoints.xy[0].cpu().numpy()
        if len(keypoints) >= 16:
            # 2. Extract joints (using your Squat/Curl logic)
            hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
            angle = calculate_angle(hip, knee, ankle)
            
            # 3. Apply your state tracking logic
            if angle > 160: stage = "down"
            if angle < 90 and stage == "down":
                stage = "up"
                counter += 1
                
            # 4. Draw the angle on the video screen so you can see it
            cv2.putText(img, f"Reps: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Angle: {int(angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- START THE CAMERA SECURELY ---
webrtc_streamer(
    key="physioform-camera",
    video_frame_callback=video_frame_callback,
    # This specific configuration helps the cloud server securely talk to your mobile browser
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} 
)
