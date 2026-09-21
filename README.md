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

## Setup

```bash
# Activate the project venv (same venv as the existing system)
.\venv\Scripts\activate       # Windows

# Dependencies already installed if you've run facerecg-main or step3/step4 before.
# If not:
pip install -r facerecg-main/requirements.txt
pip install ultralytics
```

Make sure you have:
1. **A trained PPE model** at `model/best.pt`  
   (or trained via `step3_train_medical_ppe.py`).
2. **At least one enrolled staff member** in the face recognition database  
   (run `streamlit run facerecg-main/app/Home.py` and register staff first).

---

## Running

### Unified monitor (recommended)
```bash
python unified_entry_monitor.py
```

Optional flags:
```bash
python unified_entry_monitor.py --camera-index 0 --threshold 0.62 --cooldown 15
```

### Streamlit admin dashboard
```bash
streamlit run facerecg-main/app/Home.py
```

The dashboard now shows:
- Staff registration / authorization management
- Entry logs with **PPE status** and **missing PPE** columns
- PPE compliance rate metrics

### Original standalone scripts (still work independently)
```bash
python facerecg-main/app/entry_monitor.py          # face recognition only
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
