# Unified Face Recognition + PPE Detection Entry System

A single-camera entry gate that **identifies staff by face** and **verifies their PPE compliance** — both from one webcam, in one process.

Built by integrating:
- **Face Recognition** — `facerecg-main/` (YuNet detector + FaceNet InceptionResnetV1 embeddings, open-set cosine matching)
- **Medical PPE Detection** — `step4_compliance_arbiter_medical_ppe.py` (YOLO Medical PPE model, temporal validation state machine)

---

## How It Works

```
Webcam frame
    │
    ├─► Face Detection Thread (background)
    │       YuNet → FaceNet → cosine match → AUTHORIZED / DENIED / UNKNOWN
    │
    └─► YOLO PPE Detection (main thread)
            ultralytics predict → temporal history → validated set
    │
    ▼
  State Machine
    IDLE ──(AUTHORIZED face)──► EVALUATING_PPE
                                    │
            (all PPE validated) ──► GRANTED ──(5s)──► IDLE
            (10s timeout)       ──► VIOLATION         IDLE
    │
    ▼
  SQLite entry_logs (identity + PPE status) + actuator flags printed to console
```

Only `AUTHORIZED` (enrolled + authorized) staff trigger PPE evaluation. Unknown or deauthorized faces stay in IDLE.

---

## System Requirements for Implementation
* **Operating System**: Windows 10/11 (64-bit) or Linux (Ubuntu 20.04/22.04 LTS). *(Note: Windows 7 is not supported due to modern Python 3.10+ and PyTorch/OpenCV runtime requirements).*
* **Python**: Python 3.10 – 3.13 (64-bit).
* **Hardware**:
  * **CPU-only mode**: Quad-Core Intel/AMD CPU, 4 GB RAM.
  * **GPU mode (optional)**: NVIDIA GPU with CUDA 12.x support (e.g., GTX 1650, RTX 3050+), 8 GB RAM.
* **Camera**: Standard USB Webcam (720p or 1080p).

---

## Minimal Deployment (Running Implementation Only)

If you are transferring the project to a new target machine to **run only the implementation** (no training, datasets, or test suites required), follow these steps:

### 1. Transfer the Minimal File Bundle
Copy only the following files and folders to your target system:
```text
project-root/
├── unified_entry_monitor.py       # Main camera capture & inference pipeline
├── unified_config.py              # Centralized configuration & thresholds
├── unified_state_machine.py       # Access arbitration finite state machine
├── ppe_filter.py                  # Bounding box constraints & glove IoU logic
├── face_tracker.py                # Bounding box smoothing & decision hysteresis
├── requirements_run.txt           # Minimal runtime package list
├── model/
│   └── best.pt                    # Pretrained YOLO PPE weights
└── facerecg-main/
    ├── app/
    │   ├── face_engine.py         # YuNet detection + FaceNet embeddings
    │   ├── db.py                  # SQLite database interface
    │   ├── config.py              # Face engine parameters
    │   ├── schema.sql             # Database schema
    │   └── models/
    │       └── face_detection_yunet_2023mar.onnx  # YuNet face detector model
    └── data_store/
        └── face_recognition.db    # Enrolled staff database & embeddings
```

### 2. Create and Activate Virtual Environment
Open PowerShell or Command Prompt in the project root:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Runtime Dependencies

Choose either **Option A (CPU-Only)** or **Option B (GPU Accelerated)**:

#### Option A: CPU-Only (Recommended for PCs/Laptops without dedicated NVIDIA GPU)
```bash
# 1. Install lightweight CPU PyTorch build (~200 MB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install core runtime packages (OpenCV, Ultralytics, NumPy, Pandas, Pillow, PyYAML)
pip install -r requirements_run.txt

# 3. Install facenet-pytorch with --no-deps (prevents pip from downgrading numpy on Python 3.13)
pip install facenet-pytorch --no-deps
```

#### Option B: NVIDIA GPU (With CUDA 12.4 acceleration)
```bash
# 1. Install CUDA 12.4 PyTorch build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install core runtime packages
pip install -r requirements_run.txt

# 3. Install facenet-pytorch with --no-deps
pip install facenet-pytorch --no-deps
```

---

## Full Development Setup (Optional)
If you want to train models, run the Streamlit admin dashboard, or execute the automated pytest suite:
```bash
pip install -r requirements.txt
pip install facenet-pytorch --no-deps
```

---

## Running the Implementation

### 1. Verify Prerequisites
Before launching the gate monitor, ensure:
1. The trained PPE model exists at `model/best.pt`.
2. The SQLite database at `facerecg-main/data_store/face_recognition.db` contains at least one authorized staff profile. *(If starting fresh, run `streamlit run facerecg-main/app/Home.py` once to register staff members).*

### 2. Start the Unified Entry Monitor
```bash
python unified_entry_monitor.py
```

### Optional Command-Line Arguments:
```bash
# Custom webcam index (default: 0)
python unified_entry_monitor.py --camera-index 1

# Custom face matching cosine similarity threshold (default: 0.50)
python unified_entry_monitor.py --threshold 0.55

# Custom cooldown period between repeated logs for the same person (default: 15s)
python unified_entry_monitor.py --cooldown 20
```

### Controls & HUD Display:
* **Live Video Window**: Displays real-time camera feed with:
  * **Face Bounding Box**: Green (`AUTHORIZED`), Red (`DENIED`), or Orange (`UNKNOWN`).
  * **PPE Bounding Boxes**: Mask (Red), Gloves (Green), Goggles (Cyan), Coverall (Orange).
  * **HUD Header**: Current FSM state (`IDLE`, `EVALUATING_PPE`, or `GRANTED`), recognized staff name, and evaluation countdown timer.
  * **Bottom HUD**: List of validated PPE items detected over the rolling temporal window.
* **Exit**: Press `q` or `ESC` in the video window to safely stop worker threads and close the camera.

---

## Standalone Tools & Dashboard (Optional)

### Streamlit Admin Dashboard
```bash
streamlit run facerecg-main/app/Home.py
```
Provides staff registration, authorization management, audit log inspection, and PPE compliance statistics.

### Standalone Inference Modules
```bash
python facerecg-main/app/entry_monitor.py          # Face recognition only
python step4_compliance_arbiter_medical_ppe.py     # PPE detection only
```

---

## Configuration

All tunables are in [`unified_config.py`](unified_config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MANDATED_SET` | `{'mask'}` | PPE items that must all be validated |
| `EVAL_TIMEOUT` | `10.0 s` | Time window to show all required PPE |
| `HISTORY_LEN` | `100` | Rolling frame window for temporal PPE validation |
| `MIN_DETECT_COUNT` | `5` | Frames a PPE item must appear in to be "validated" |
| `GRANTED_HOLD_SECONDS` | `5.0 s` | How long ACCESS GRANTED is displayed |
| `MATCH_THRESHOLD` | `0.62` | Cosine similarity threshold for face matching |
| `CAMERA_INDEX` | `0` | Webcam index |

---

## Architecture

| File | Purpose |
|------|---------|
| [`unified_config.py`](unified_config.py) | All tunables — face recog + PPE + combined |
| [`unified_state_machine.py`](unified_state_machine.py) | Pure state machine (IDLE → EVALUATING_PPE → GRANTED/VIOLATION). No OpenCV dependency — fully unit-testable. |
| [`unified_entry_monitor.py`](unified_entry_monitor.py) | Camera loop: face worker thread + YOLO PPE + rendering + DB logging |
| [`facerecg-main/app/`](facerecg-main/app/) | Face engine, DB layer, Streamlit UI (unchanged) |
| [`step4_compliance_arbiter_medical_ppe.py`](step4_compliance_arbiter_medical_ppe.py) | Original standalone PPE arbiter (unchanged) |
| [`tests/`](tests/) | pytest unit tests (51 tests, no camera required) |

---

## Running Tests

```bash
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: **51 passed** — covers config, DB schema/migration, and all state machine transitions.

---

## Console Output

```
[INFO] Unified Entry Monitor running. Press 'q' or ESC to quit.
[INFO] Mandated PPE: {'mask'}

[GRANTED] Alice — all PPE compliant.
>> ACTUATOR: door_open = True

[VIOLATION] Bob — missing PPE: mask
>> ACTUATOR: dispense_mask = True
```
