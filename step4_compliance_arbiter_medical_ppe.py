import cv2
import time
from collections import deque
from ultralytics import YOLO
import os

# ==========================================
# CONFIGURATION
# ==========================================
# All four Medical PPE classes are mandated by default.
# Remove items from this set if you only want to enforce a subset.
# MANDATED_SET = {'coverall', 'gloves', 'goggles', 'mask'}
MANDATED_SET = {'mask'}

# Temporal Validation Settings
HISTORY_LEN = 100     # Number of frames to keep in history
MIN_DETECT_COUNT = 5  # Item must be in at least this many frames to be 'Validated'
EVAL_TIMEOUT = 10.0   # 10 seconds window to show all required PPE

DEFAULT_CONF = 0.5    # Default confidence threshold for all classes
# Per-class overrides (higher threshold = fewer false positives)
CLASS_CONF = {
    1:0.65,
    2: 0.85,          # Goggles — raised to reduce false positives on spectacles
}

# YOLO Class Mapping (Medical PPE dataset)
# 0: Coverall, 1: Gloves, 2: Goggles, 3: Mask
CLASS_NAMES = {
    0: 'coverall',
    1: 'gloves',
    2: 'goggles',
    3: 'mask',
}

MODEL_PATH = os.path.join("runs", "detect", "runs", "detect", "ppe_medical", "weights", "best.pt")


def main():
    print("==========================================")
    print(" YOLO PPE Engine - Medical PPE Arbiter")
    print("==========================================")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Could not find {MODEL_PATH}.")
        print("        Did step3_train_medical_ppe.py finish successfully?")
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

        # Parse detections in this frame (with per-class confidence filtering)
        current_frame_detections = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                min_conf = CLASS_CONF.get(cls_id, DEFAULT_CONF)
                if cls_id in CLASS_NAMES and conf >= min_conf:
                    current_frame_detections.add(CLASS_NAMES[cls_id])

        # Add to rolling history
        history.append(current_frame_detections)

        # 2. Temporal Validation
        # Count how many times each class appeared in the last HISTORY_LEN frames
        counts = {name: 0 for name in CLASS_NAMES.values()}
        for dets in history:
            for d in dets:
                counts[d] += 1

        # A class is "temporally validated" if it meets the MIN_DETECT_COUNT
        validated_set = {cls for cls, count in counts.items() if count >= MIN_DETECT_COUNT}

        # 3. State Machine Logic
        # Draw only boxes that pass the per-class confidence threshold
        annotated_frame = frame.copy()
        BOX_COLORS = {
            0: (255, 165, 0),   # Coverall - orange
            1: (0, 255, 0),     # Gloves   - green
            2: (255, 255, 0),   # Goggles  - cyan
            3: (0, 0, 255),     # Mask     - red
        }
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                min_conf = CLASS_CONF.get(cls_id, DEFAULT_CONF)
                if cls_id in CLASS_NAMES and conf >= min_conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = BOX_COLORS.get(cls_id, (255, 255, 255))
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if state == "IDLE":
            # If we see ANY mandated item, someone is likely in frame. Start evaluation.
            if len(validated_set.intersection(MANDATED_SET)) > 0:
                print("\n[EVENT] Subject detected. Starting 10-second compliance window...")
                state = "EVALUATING"
                eval_start_time = time.time()

        elif state == "EVALUATING":
            elapsed = time.time() - eval_start_time

            # Check compliance
            missing = MANDATED_SET - validated_set

            # Draw UI Timer
            cv2.putText(annotated_frame, f"Evaluating: {EVAL_TIMEOUT - elapsed:.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.putText(annotated_frame, f"Missing: {', '.join(missing) if missing else 'None'}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if missing else (0, 255, 0), 2)

            if len(missing) == 0:
                print("\n[SUCCESS] Full compliance reached!")
                print(">> ACTUATOR: door_open = True")
                state = "GRANTED"
                eval_start_time = time.time()

            elif elapsed > EVAL_TIMEOUT:
                print(f"\n[VIOLATION] 10 seconds expired. Missing PPE: {missing}")
                for m in missing:
                    print(f">> ACTUATOR: dispense_{m} = True")

                # Reset to IDLE after dispensing
                history.clear()
                state = "IDLE"

        elif state == "GRANTED":
            cv2.putText(annotated_frame, "ACCESS GRANTED", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            # Hold door open for 5 seconds
            if time.time() - eval_start_time > 5.0:
                print("[INFO] Door closed. Returning to IDLE.")
                history.clear()
                state = "IDLE"

        # 4. Display Detections
        val_text = f"Validated: {', '.join(sorted(validated_set)) if validated_set else 'None'}"
        cv2.putText(annotated_frame, val_text, (20, annotated_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw Critical Zone label at top right
        zone_text = "Critical Zone"
        (text_w, text_h), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(annotated_frame, zone_text, (annotated_frame.shape[1] - text_w - 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Medical PPE Arbiter", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
