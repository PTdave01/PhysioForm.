print("--- PHYSIOFORM: REP TRACKING SIMULATION ---")

# We simulate the YOLO model outputting these elbow angles frame-by-frame.
# The user starts at 160 (arms down), curls up to 30, goes back down, and curls up again.
frame_angles = [160, 140, 110, 80, 45, 30, 45, 90, 130, 155, 120, 80, 35, 80, 160]

# --- STATE TRACKING VARIABLES ---
counter = 0
stage = None 

for frame_number, angle in enumerate(frame_angles):
    
    # 1. EXTENSION PHASE: If the angle is greater than 150 degrees, the arm is straight down.
    if angle > 150:
        stage = "down"
        
    # 2. FLEXION PHASE: If the angle is less than 40 degrees AND the arm was previously down, it's a rep!
    if angle < 40 and stage == "down":
        stage = "up"       # Update the state so we don't double-count the same rep
        counter += 1       # Add +1 to the total reps
        print(f"\n✅ PERFECT REP DETECTED! Total: {counter}\n")
        
    # Print the live tracking data
    print(f"Frame {frame_number} | Current Angle: {angle}° | Stage: {stage} | Total Reps: {counter}")

print("\n--- SIMULATION COMPLETE ---")
