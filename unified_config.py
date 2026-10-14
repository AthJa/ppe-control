"""Unified configuration for the combined Face Recognition + PPE Detection system.

Merges tunables from:
  - facerecg-main/app/config.py  (face recognition)
  - step4_compliance_arbiter_medical_ppe.py  (PPE detection)

All paths are resolved relative to this file's location (project root).
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACERECG_APP_DIR = os.path.join(PROJECT_ROOT, "facerecg-main", "app")

# Face recognition data store (shared with facerecg-main)
DATA_STORE_DIR = os.path.join(PROJECT_ROOT, "facerecg-main", "data_store")
FACE_DB_PATH = os.path.join(DATA_STORE_DIR, "face_recognition.db")
FACE_SCHEMA_PATH = os.path.join(FACERECG_APP_DIR, "schema.sql")

# Face detection model (YuNet ONNX)
YUNET_MODEL_PATH = os.path.join(FACERECG_APP_DIR, "models", "face_detection_yunet_2023mar.onnx")

# Snapshots
SNAPSHOTS_DIR = os.path.join(DATA_STORE_DIR, "snapshots")
ENTRY_SNAPSHOTS_DIR = os.path.join(SNAPSHOTS_DIR, "entry")

# PPE YOLO model
PPE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best.pt")

# ---------------------------------------------------------------------------
# Face Recognition Tunables (from facerecg-main/app/config.py)
# ---------------------------------------------------------------------------

# Embedding
EMBEDDING_DIM = 512
MATCH_THRESHOLD = 0.50      # calibrated threshold for stable live webcam matching


# Entry monitor
CAMERA_INDEX = 0
LOG_COOLDOWN_SECONDS = 15
DETECTION_DOWNSCALE = 0.5   # resize factor applied before face detection; boxes scaled back up

# Detector (YuNet)
DETECTOR_SCORE_THRESHOLD = 0.5   # lowered from 0.9 — occluded (mask/cap) faces score lower
YUNET_NMS_THRESHOLD = 0.3

# Enrollment
MIN_ENROLL_IMAGES = 3
BURST_CAPTURE_COUNT = 8
BURST_CAPTURE_INTERVAL_SEC = 1.2

# Face Tracking & Anti-Flicker (smoothing, persistence, hysteresis)
FACE_TRACK_MAX_LOST_FRAMES = 10   # persistence frames across momentary detector misses
FACE_SMOOTHING_ALPHA = 0.7       # EMA weight for box coordinates (higher = responsive, lower = smoother)
FACE_HYSTERESIS_MARGIN = 0.06    # allowed similarity dip below threshold while maintaining AUTHORIZED
FACE_HYSTERESIS_FRAMES = 5       # consecutive low-similarity frames needed to revert to UNKNOWN


# ---------------------------------------------------------------------------
# PPE Detection Tunables (from step4_compliance_arbiter_medical_ppe.py)
# ---------------------------------------------------------------------------

# All mandated PPE classes — adjust to enforce a different subset
MANDATED_SET = {'mask'}
# MANDATED_SET = {'coverall', 'gloves', 'goggles', 'mask'}

# Temporal validation
HISTORY_LEN = 100       # number of frames to keep in rolling history
MIN_DETECT_COUNT = 10    # item must appear in at least this many frames to be "validated"
EVAL_TIMEOUT = 10.0     # seconds to present all required PPE

# YOLO confidence
DEFAULT_CONF = 0.6
CLASS_CONF = {
    1: 0.65,
    2: 0.85,    # Goggles — raised to reduce false positives on spectacles
}

# YOLO class mapping (Medical PPE dataset)
# 0: Coverall, 1: Gloves, 2: Goggles, 3: Mask
CLASS_NAMES = {
    0: 'coverall',
    1: 'gloves',
    2: 'goggles',
    3: 'mask',
}

# Bounding box colors per PPE class (BGR)
PPE_BOX_COLORS = {
    0: (255, 165, 0),   # Coverall — orange
    1: (0, 255, 0),     # Gloves   — green
    2: (255, 255, 0),   # Goggles  — cyan
    3: (0, 0, 255),     # Mask     — red
}

# Maximum boxes to display per class at all times
MAX_BOXES_PER_CLASS = {
    0: 1,  # coverall — max 1 box
    1: 2,  # gloves   — max 2 boxes
    2: 1,  # goggles  — max 1 box
    3: 1,  # mask     — max 1 box
    'coverall': 1,
    'gloves': 2,
    'goggles': 1,
    'mask': 1,
}

# Maximum allowed IoU between two glove boxes to consider them non-overlapping
GLOVE_OVERLAP_IOU_THRESHOLD = 0.1

# ---------------------------------------------------------------------------

# Combined System Tunables
# ---------------------------------------------------------------------------

GRANTED_HOLD_SECONDS = 5.0   # how long to hold the GRANTED state before returning to IDLE
