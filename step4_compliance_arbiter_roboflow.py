import cv2
import time
from collections import deque
from ultralytics import YOLO
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Configure your mandated set here. 
MANDATED_SET = {'mask', 'cap', 'glove'}

# Temporal Validation Settings
HISTORY_LEN = 15      # Number of frames to keep in history
MIN_DETECT_COUNT = 5  # Item must be in at least this many frames to be 'Validated'
EVAL_TIMEOUT = 10.0   # 10 seconds window to show all required PPE

# YOLO Class Mapping (based on the Roboflow dataset classes)
# 0: Coverall, 1: Gloves, 2: Goggles, 3: Mask
# We map Coverall to 'cap' since the dataset doesn't have a cap class.
CLASS_NAMES = {
    0: 'cap',
    1: 'glove',
    2: None,      # We ignore goggles for now
    3: 'mask'
}

MODEL_PATH = r"runs\detect\runs\detect\ppe_engine_roboflow\weights\best.pt"

def main():
    print("==========================================")
    print(" YOLO PPE Engine - Roboflow Local Arbiter")
    print("==========================================")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Could not find {MODEL_PATH}. Did the training finish successfully?")
        return

    print(f"[INFO] Loading Model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    print(f"[INFO] Mandated PPE Set: {MANDATED_SET}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # State Variables
    history = deque(maxlen=HISTORY_LEN)
    state = "IDLE"  # IDLE, EVALUATING, GRANTED
    eval_start_time = 0.0

    print("[INFO] Starting camera feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Run YOLO Inference
        results = model.predict(frame, verbose=False, conf=0.5)
        
        # Parse detections in this frame
        current_frame_detections = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in CLASS_NAMES and CLASS_NAMES[cls_id] is not None:
                    current_frame_detections.add(CLASS_NAMES[cls_id])
                    
        # Add to rolling history
        history.append(current_frame_detections)
        
        # 2. Temporal Validation
        counts = {'mask': 0, 'glove': 0, 'cap': 0}
        for dets in history:
            for d in dets:
                counts[d] += 1
                
        # A class is "temporally validated" if it meets the MIN_DETECT_COUNT
        validated_set = {cls for cls, count in counts.items() if count >= MIN_DETECT_COUNT}

        # 3. State Machine Logic
        annotated_frame = results[0].plot() # get annotated image

        if state == "IDLE":
            # If we see ANY mandated item, someone is likely in frame. Start evaluation window.
            if len(validated_set.intersection(MANDATED_SET)) > 0:
                print("\n[EVENT] Subject detected. Starting 10-second compliance window...")
                state = "EVALUATING"
                eval_start_time = time.time()
                
        elif state == "EVALUATING":
            elapsed = time.time() - eval_start_time
            
            # Check if they have met full compliance
            missing = MANDATED_SET - validated_set
            
            # Draw UI Timer
            cv2.putText(annotated_frame, f"Evaluating: {10.0 - elapsed:.1f}s", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(annotated_frame, f"Missing: {', '.join(missing) if missing else 'None'}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if missing else (0, 255, 0), 2)
            
            if len(missing) == 0:
                print("\n[SUCCESS] Full compliance reached!")
                print(">> ACTUATOR: door_open = True")
                state = "GRANTED"
                eval_start_time = time.time() # repurpose timer for door open duration
                
            elif elapsed > EVAL_TIMEOUT:
                # Timeout reached! Calculate Set Difference and dispense
                print(f"\n[VIOLATION] 10 seconds expired. Missing PPE: {missing}")
                for m in missing:
                    print(f">> ACTUATOR: dispense_{m} = True")
                
                # Reset to IDLE after dispensing
                history.clear()
                state = "IDLE"
                
        elif state == "GRANTED":
            cv2.putText(annotated_frame, "ACCESS GRANTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            # Hold the door open for 5 seconds before resetting
            if time.time() - eval_start_time > 5.0:
                print("[INFO] Door closed. Returning to IDLE.")
                history.clear()
                state = "IDLE"

        # 4. Display Detections
        val_text = f"Validated: {', '.join(validated_set)}"
        cv2.putText(annotated_frame, val_text, (20, annotated_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Roboflow PPE Arbiter", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
