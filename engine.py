import numpy as np
from ultralytics import YOLO

# --- THE TRIGONOMETRY ENGINE ---
def calculate_angle(a, b, c):
    """
    Calculates the internal angle between three (x, y) coordinate points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

print("Loading YOLOv8 Pose Model...")
model = YOLO('yolov8n-pose.pt') 

print("Model loaded! Analyzing test image...")
results = model('https://ultralytics.com/images/bus.jpg')

print("\n--- PHYSIOFORM: JOINT ANGLE RESULTS ---")
for r in results:
    # Extract the keypoints for the first person detected
    keypoints = r.keypoints.xy[0].cpu().numpy()
    
    # We need to make sure the model detected up to the ankle (index 15)
    if len(keypoints) >= 16:
        
        # --- 1. BICEP CURL MATH ---
        # Isolate Left Arm (Shoulder: 5, Elbow: 7, Wrist: 9)
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        arm_angle = calculate_angle(shoulder, elbow, wrist)
        
        # --- 2. SQUAT MATH ---
        # Isolate Left Leg (Hip: 11, Knee: 13, Ankle: 15)
        hip = keypoints[11]
        knee = keypoints[13]
        ankle = keypoints[15]
        leg_angle = calculate_angle(hip, knee, ankle)
        
        # --- DISPLAY THE RESULTS ---
        print(f"BICEP CURL - Left Elbow Angle : {arm_angle:.2f} degrees")
        print(f"SQUAT      - Left Knee Angle  : {leg_angle:.2f} degrees")
        print(f"-------------------------------------------------------")
