import cv2
import time
import base64
import requests
import urllib.parse
from collections import deque
import os
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================
# Roboflow API configuration
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY') # <--- REPLACE WITH YOUR ROBOFLOW API KEY
MODEL_ENDPOINT = "medical-ppe-o6ot6-iyays/1" # Your forked project!

MANDATED_SET = {'mask', 'cap', 'glove'}

# Temporal Validation Settings
HISTORY_LEN = 15      
MIN_DETECT_COUNT = 5  
EVAL_TIMEOUT = 10.0   

# We map roboflow classes to our standardized set
# E.g., if roboflow calls it "medical mask", we map it to "mask"
CLASS_MAP = {
    'mask': 'mask', 'face_mask': 'mask', 'medical mask': 'mask', 'with_mask': 'mask',
    'glove': 'glove', 'gloves': 'glove', 'medical_glove': 'glove',
    'cap': 'cap', 'head_cover': 'cap', 'hair_cover': 'cap', 'mob_cap': 'cap', 'coverall': 'cap'
}

# ==========================================
# STATE & BUFFERS
# ==========================================
detection_history = deque(maxlen=HISTORY_LEN)

STATE = 'IDLE'
eval_start_time = 0
validated_in_window = set()

def get_roboflow_predictions(frame):
    """Sends the frame to Roboflow Inference API and returns predictions."""
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Please set your ROBOFLOW_API_KEY in the script.")
        return []
        
    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode("ascii")

    # Construct the URL
    upload_url = "".join([
        "https://detect.roboflow.com/",
        MODEL_ENDPOINT,
        "?api_key=",
        ROBOFLOW_API_KEY,
        "&format=json"
    ])

    try:
        resp = requests.post(
            upload_url,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=5 # Prevent hanging forever
        )
        data = resp.json()
        if "predictions" in data:
            return data["predictions"]
        else:
            print(f"[API ERROR] {data}")
            return []
    except Exception as e:
        print(f"[API EXCEPTION] {e}")
        return []

def main():
    global STATE, eval_start_time, validated_in_window
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("========================================")
    print(" Online PPE Arbiter (Roboflow Universe) ")
    print("========================================")
    print(f"Mandated Set: {MANDATED_SET}")
    
    # To prevent API rate limits and severe lag, we only process 2 frames per second
    last_api_call = 0
    api_cooldown = 0.5 
    
    latest_preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        
        # Call API if cooldown has passed
        if current_time - last_api_call > api_cooldown:
            latest_preds = get_roboflow_predictions(frame)
            last_api_call = current_time
            
            # Extract classes we care about
            current_detected = set()
            for p in latest_preds:
                # Roboflow usually returns "class"
                class_name = p.get('class', '').lower()
                mapped_name = CLASS_MAP.get(class_name)
                
                if mapped_name and mapped_name in MANDATED_SET:
                    current_detected.add(mapped_name)
            
            # Add to temporal history
            detection_history.append(current_detected)

        # Count occurrences in history
        counts = {item: 0 for item in MANDATED_SET}
        for det_set in detection_history:
            for item in det_set:
                counts[item] += 1

        # Determine currently Validated items
        currently_validated = set()
        for item, count in counts.items():
            if count >= MIN_DETECT_COUNT:
                currently_validated.add(item)
                validated_in_window.add(item) 

        # ==========================
        # STATE MACHINE LOGIC
        # ==========================
        if STATE == 'IDLE':
            if len(currently_validated) > 0:
                print("\n[EVENT] PPE detected. Starting 10-second evaluation window...")
                STATE = 'EVALUATING'
                eval_start_time = current_time
                validated_in_window = set(currently_validated)
                
        elif STATE == 'EVALUATING':
            elapsed = current_time - eval_start_time
            cv2.putText(frame, f"Evaluating: {10.0 - elapsed:.1f}s", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # Fast track success
            if MANDATED_SET.issubset(validated_in_window):
                print(f"[SUCCESS] All mandated PPE detected: {validated_in_window}")
                print(">> ACTUATOR: door_open = True")
                STATE = 'GRANTED'
                
            # Timeout
            elif elapsed > EVAL_TIMEOUT:
                missing = MANDATED_SET - validated_in_window
                print(f"[TIMEOUT] Evaluation finished. Missing items: {missing}")
                
                # Actuator flags
                if 'mask' in missing:
                    print(">> ACTUATOR: dispense_mask = True")
                if 'cap' in missing:
                    print(">> ACTUATOR: dispense_cap = True")
                if 'glove' in missing:
                    print(">> ACTUATOR: dispense_glove = True")
                
                STATE = 'IDLE'
                detection_history.clear()
                validated_in_window.clear()

        elif STATE == 'GRANTED':
            cv2.putText(frame, "ACCESS GRANTED", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Reset after a few seconds of empty frame (to allow the person to walk through)
            if len(currently_validated) == 0 and sum(len(d) for d in detection_history) == 0:
                print("[EVENT] Area cleared. Resetting to IDLE.")
                STATE = 'IDLE'

        # ==========================
        # UI OVERLAYS
        # ==========================
        cv2.putText(frame, f"State: {STATE}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw bounding boxes from latest prediction
        for p in latest_preds:
            class_name = p.get('class', '')
            mapped = CLASS_MAP.get(class_name.lower())
            
            # Only draw if it's a class we care about
            if mapped in MANDATED_SET:
                x = int(p.get('x', 0) - p.get('width', 0)/2)
                y = int(p.get('y', 0) - p.get('height', 0)/2)
                w = int(p.get('width', 0))
                h = int(p.get('height', 0))
                conf = p.get('confidence', 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{mapped} {conf:.2f}", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        y_offset = 100
        for item in MANDATED_SET:
            color = (0, 255, 0) if item in currently_validated else (0, 0, 255)
            cv2.putText(frame, f"{item.upper()}: {'[X]' if item in currently_validated else '[ ]'}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        cv2.imshow("Roboflow PPE Arbiter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
