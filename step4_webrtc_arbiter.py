import os
import cv2
import time
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

if not ROBOFLOW_API_KEY:
    raise ValueError("Please set ROBOFLOW_API_KEY in your .env file!")

# Temporal Validation Settings
MANDATED_SET = {'mask', 'cap', 'glove'}
HISTORY_LEN = 15      # Frames to keep in history
MIN_DETECT_COUNT = 5  # Item must be in at least this many frames to be 'Validated'
EVAL_TIMEOUT = 10.0   # 10 seconds window

# Class mapping (mapping 'coverall' to 'cap' since your dataset lacks a cap class)
CLASS_MAP = {
    'mask': 'mask', 'face_mask': 'mask', 'medical mask': 'mask', 'with_mask': 'mask',
    'glove': 'glove', 'gloves': 'glove', 'medical_glove': 'glove',
    'cap': 'cap', 'head_cover': 'cap', 'hair_cover': 'cap', 'mob_cap': 'cap', 'coverall': 'cap'
}

# State
history = []
STATE = "EVALUATING"
eval_start_time = time.time()
validated_in_window = set()

# Initialize client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    processing_timeout=3600,             
    requested_plan="webrtc-gpu-medium",  
    requested_region="us"                
)

print("Starting WebRTC Stream... This may take a moment to connect.")

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="medical-ppe-vmedical-ppe-o6ot6-iyays-1-yolo11n-t1-logic", # Your workflow!
    workspace="aaa-2kbod",
    image_input="image",
    config=config
)

@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("WebRTC PPE Arbiter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    global STATE, eval_start_time, validated_in_window, history

    if STATE == "COMPLIANT":
        return

    current_time = time.time()
    
    # Reset window if timeout reached
    if current_time - eval_start_time > EVAL_TIMEOUT:
        print("\n[TIMEOUT] 10 seconds elapsed. Restarting evaluation.")
        eval_start_time = current_time
        validated_in_window.clear()
        history.clear()

    # Parse predictions
    # The output format for workflows is deeply nested depending on the workflow blocks.
    # Usually, it's under data['predictions'] or similar. We will try to extract class names safely.
    current_frame_detections = set()
    
    # Print the raw data once just to see the structure if needed
    if metadata.frame_id == 1:
        print("Raw WebRTC Output Format:", data)

    # A typical Roboflow Workflow Object Detection output looks like:
    # {'predictions': [{'class': 'Coverall', ...}]} OR {'output': [{'class': 'Mask', ...}]}
    
    # We will search the entire dict recursively for any key named 'class'
    def extract_classes(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'class' and isinstance(v, str):
                    cls_id = v.lower()
                    if cls_id in CLASS_MAP:
                        current_frame_detections.add(CLASS_MAP[cls_id])
                else:
                    extract_classes(v)
        elif isinstance(d, list):
            for item in d:
                extract_classes(item)
                
    extract_classes(data)

    history.append(current_frame_detections)
    if len(history) > HISTORY_LEN:
        history.pop(0)

    # Count how many times each class appeared in the last HISTORY_LEN frames
    counts = {item: 0 for item in MANDATED_SET}
    for frame_dets in history:
        for item in frame_dets:
            if item in counts:
                counts[item] += 1

    # Update validated items
    for item, count in counts.items():
        if count >= MIN_DETECT_COUNT:
            validated_in_window.add(item)

    missing = MANDATED_SET - validated_in_window

    print(f"Frame {metadata.frame_id} | Window: {10.0 - (current_time - eval_start_time):.1f}s | Validated: {list(validated_in_window)} | Missing: {list(missing)}")

    if not missing:
        print("\n==========================================")
        print(" [ACTUATOR SIGNAL] -> DOOR OPEN")
        print(" FULL COMPLIANCE REACHED! ")
        print("==========================================")
        STATE = "COMPLIANT"

# Run the session (blocks until closed)
session.run()
cv2.destroyAllWindows()
