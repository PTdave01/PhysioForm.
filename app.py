import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="PhysioForm")
st.title("PhysioForm: Clinical Movement Tracker")

import mediapipe as mp
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: 
        angle = 360 - angle
    return angle

exercise = st.selectbox("Select Exercise to Track", ["Squat", "Bicep Curl"])

class PhysioProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.exercise_type = "Squat"
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if self.exercise_type == "Squat":
                pt1 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0]]
                pt2 = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * img.shape[0]]
                pt3 = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * img.shape[0]]
                down_thresh, up_thresh = 160, 90
            else:
                pt1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]]
                pt2 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]]
                pt3 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * img.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * img.shape[0]]
                down_thresh, up_thresh = 150, 40

            angle = calculate_angle(pt1, pt2, pt3)
            
            if angle > down_thresh:
                self.stage = "down"
            if angle < up_thresh and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                
            cv2.putText(img, f"{self.exercise_type.upper()} REPS: {self.counter}", 
                        (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(img, f"ANGLE: {int(angle)}", 
                        (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="physioform",
    mode=WebRtcMode.SENDRECV,  # <-- changed from "SENDRECV"
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
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.exercise_type = exercise
