"""Unified Entry Monitor — Face Recognition + Medical PPE Detection.

Single-camera loop that:
  1. Identifies the person via face recognition (background thread, same pattern
     as facerecg-main/app/entry_monitor.py).
  2. Once a person is AUTHORIZED, enters a PPE compliance evaluation window
     (YOLO Medical PPE model, temporal validation, same logic as
     step4_compliance_arbiter_medical_ppe.py).
  3. Draws face boxes + PPE boxes + HUD overlay on the same frame.
  4. Logs each outcome (identity + PPE status) to the shared SQLite database.

Usage
-----
    python unified_entry_monitor.py [--camera-index 0] [--threshold 0.62]

State Machine
-------------
    IDLE → (AUTHORIZED face) → EVALUATING_PPE
                                  │
                  (all PPE validated) → GRANTED → (5s hold) → IDLE
                  (timeout)           → VIOLATION            → IDLE
"""
import argparse
import os
import sys
import threading
import time

import cv2

# ---------------------------------------------------------------------------
# Path setup — import face recognition modules from facerecg-main/app
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FACERECG_APP = os.path.join(PROJECT_ROOT, "facerecg-main", "app")
sys.path.insert(0, FACERECG_APP)
sys.path.insert(0, PROJECT_ROOT)

import face_engine
import db

import unified_config as cfg
from unified_state_machine import UnifiedStateMachine, State

# ---------------------------------------------------------------------------
# YOLO — loaded lazily to avoid import-time cost when camera isn't available
# ---------------------------------------------------------------------------
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(cfg.PPE_MODEL_PATH)
    return _yolo_model


# ---------------------------------------------------------------------------
# Face Detection Worker (background thread)
# ---------------------------------------------------------------------------

class _FaceWorker:
    """Runs detect → embed → match on the most recent frame in a background thread.

    Identical threading pattern to facerecg-main/app/entry_monitor.py's
    _DetectionWorker, so the video preview never freezes on CPU-bound inference.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._results_lock = threading.Lock()
        self.last_faces: list = []   # list of (x1, y1, x2, y2, MatchResult)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def submit_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_id += 1

    def get_results(self) -> list:
        with self._results_lock:
            return list(self.last_faces)

    def _loop(self):
        processed_id = 0
        while not self._stop.is_set():
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
            if frame is None or frame_id == processed_id:
                time.sleep(0.01)
                continue
            processed_id = frame_id

            small = cv2.resize(frame, None,
                               fx=cfg.DETECTION_DOWNSCALE, fy=cfg.DETECTION_DOWNSCALE)
            detections = face_engine.detect_faces(small)
            scale = 1.0 / cfg.DETECTION_DOWNSCALE

            results = []
            for det in detections:
                x1, y1, x2, y2 = [int(v * scale) for v in det["box"]]
                x1, y1 = max(x1, 0), max(y1, 0)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                emb = face_engine.get_embedding(crop)
                m = face_engine.match(emb, threshold=self.threshold)
                results.append((x1, y1, x2, y2, m))

            with self._results_lock:
                self.last_faces = results


# ---------------------------------------------------------------------------
# PPE Detection (main thread)
# ---------------------------------------------------------------------------

def _detect_ppe(frame) -> tuple[set, list]:
    """Run YOLO PPE detection on a frame.

    Returns
    -------
    detected_names : set of PPE class names that passed per-class confidence
    raw_boxes      : list of (cls_id, conf, x1, y1, x2, y2) for rendering
    """
    model = _get_yolo()
    results = model.predict(frame, verbose=False, conf=cfg.DEFAULT_CONF)
    detected_names: set = set()
    raw_boxes: list = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            min_conf = cfg.CLASS_CONF.get(cls_id, cfg.DEFAULT_CONF)
            if cls_id in cfg.CLASS_NAMES and conf >= min_conf:
                detected_names.add(cfg.CLASS_NAMES[cls_id])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_boxes.append((cls_id, conf, x1, y1, x2, y2))

    return detected_names, raw_boxes


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

FACE_COLORS = {
    "AUTHORIZED": (0, 200, 0),    # green
    "DENIED":     (0, 0, 255),    # red
    "UNKNOWN":    (0, 165, 255),  # orange
}


def _draw_face_boxes(frame, face_results):
    for x1, y1, x2, y2, m in face_results:
        color = FACE_COLORS.get(m.decision, (200, 200, 200))
        label = f"{m.name or 'UNKNOWN'} ({m.decision})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _draw_ppe_boxes(frame, ppe_boxes):
    for cls_id, conf, x1, y1, x2, y2 in ppe_boxes:
        color = cfg.PPE_BOX_COLORS.get(cls_id, (255, 255, 255))
        label = f"{cfg.CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_hud(frame, sm: UnifiedStateMachine):
    h, w = frame.shape[:2]

    # --- top-left: state + person ---
    state_label = f"State: {sm.state.value}"
    cv2.putText(frame, state_label, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if sm.recognized_person:
        cv2.putText(frame, f"Person: {sm.recognized_person}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

    # --- top-right: Critical Zone label ---
    zone_text = "Critical Zone"
    (tw, _), _ = cv2.getTextSize(zone_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, zone_text, (w - tw - 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # --- EVALUATING_PPE HUD ---
    if sm.state == State.EVALUATING_PPE:
        cv2.putText(frame, f"Evaluating: {sm.hud_eval_remaining:.1f}s", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        missing_str = ', '.join(sorted(sm.hud_missing)) if sm.hud_missing else 'None'
        missing_color = (0, 0, 255) if sm.hud_missing else (0, 255, 0)
        cv2.putText(frame, f"Missing: {missing_str}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, missing_color, 2)

    # --- GRANTED overlay ---
    elif sm.state == State.GRANTED:
        cv2.putText(frame, "ACCESS GRANTED", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # --- bottom: validated PPE list ---
    val_text = f"Validated PPE: {', '.join(sorted(sm.validated_set)) or 'None'}"
    cv2.putText(frame, val_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _ensure_snapshot_dirs():
    for d in ("AUTHORIZED", "DENIED", "UNKNOWN"):
        os.makedirs(os.path.join(cfg.ENTRY_SNAPSHOTS_DIR, d), exist_ok=True)


def _save_snapshot(frame, decision: str, key) -> str | None:
    crop_dir = os.path.join(cfg.ENTRY_SNAPSHOTS_DIR, decision)
    os.makedirs(crop_dir, exist_ok=True)
    path = os.path.join(crop_dir, f"{int(time.time())}_{key}.jpg")
    cv2.imwrite(path, frame)
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(camera_index: int, threshold: float, cooldown: float) -> int:
    # Initialise DB + face models
    db.init_db()
    face_engine.load_all_embeddings()
    _ensure_snapshot_dirs()

    if not os.path.exists(cfg.PPE_MODEL_PATH):
        print(f"[ERROR] PPE model not found: {cfg.PPE_MODEL_PATH}")
        print("        Did step3_train_medical_ppe.py finish successfully?")
        return 1

    # Pre-load YOLO
    print("[INFO] Loading YOLO PPE model…")
    _get_yolo()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}", file=sys.stderr)
        return 1

    sm = UnifiedStateMachine(
        mandated_set=cfg.MANDATED_SET,
        history_len=cfg.HISTORY_LEN,
        min_detect_count=cfg.MIN_DETECT_COUNT,
        eval_timeout=cfg.EVAL_TIMEOUT,
        granted_hold_seconds=cfg.GRANTED_HOLD_SECONDS,
    )

    face_worker = _FaceWorker(threshold)
    face_worker.start()

    last_logged: dict = {}   # key → last log timestamp (debounce)

    print("[INFO] Unified Entry Monitor running. Press 'q' or ESC to quit.")
    print(f"[INFO] Mandated PPE: {cfg.MANDATED_SET}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # --- face recognition (result from last background pass) ---
            face_worker.submit_frame(frame)
            face_results = face_worker.get_results()

            # Pick the best (highest similarity) recognized face for state machine
            best_face_decision = "UNKNOWN"
            best_face_name = None
            best_face_staff_id = None
            best_face_sim = 0.0

            for _, _, _, _, m in face_results:
                if m.decision == "AUTHORIZED" and m.similarity > best_face_sim:
                    best_face_decision = m.decision
                    best_face_name = m.name
                    best_face_staff_id = m.staff_id
                    best_face_sim = m.similarity

            # --- PPE detection (main thread) ---
            ppe_names, ppe_boxes = _detect_ppe(frame)

            # --- update state machine ---
            sm.update(
                face_decision=best_face_decision,
                face_name=best_face_name,
                ppe_detections=ppe_names,
                face_staff_id=best_face_staff_id,
                face_similarity=best_face_sim,
            )

            # --- handle pending log entry ---
            log = sm.consume_log()
            if log:
                now = time.time()
                key = log.get('staff_id') or "UNKNOWN"
                if now - last_logged.get(key, 0) >= cooldown:
                    last_logged[key] = now
                    snapshot_path = _save_snapshot(frame, log['decision'], key)
                    db.insert_entry_log(
                        staff_id=log['staff_id'],
                        matched_name=log['name'],
                        similarity=log['similarity'],
                        decision=log['decision'],
                        snapshot_path=snapshot_path,
                        ppe_status=log['ppe_status'],
                        ppe_missing=log.get('ppe_missing') or None,
                    )
                    if log['ppe_status'] == 'COMPLIANT':
                        print(f"\n[GRANTED] {log['name']} — all PPE compliant.")
                        print(">> ACTUATOR: door_open = True")
                    else:
                        missing = log.get('ppe_missing', '')
                        print(f"\n[VIOLATION] {log['name']} — missing PPE: {missing}")
                        for item in missing.split(','):
                            item = item.strip()
                            if item:
                                print(f">> ACTUATOR: dispense_{item} = True")

            # --- rendering ---
            _draw_face_boxes(frame, face_results)
            _draw_ppe_boxes(frame, ppe_boxes)
            _draw_hud(frame, sm)

            cv2.imshow("Unified Entry Monitor — press q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        face_worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Face Recognition + PPE Detection")
    parser.add_argument("--camera-index", type=int, default=cfg.CAMERA_INDEX,
                        help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=cfg.MATCH_THRESHOLD,
                        help="Face match cosine similarity threshold (default: 0.62)")
    parser.add_argument("--cooldown", type=float, default=cfg.LOG_COOLDOWN_SECONDS,
                        help="Seconds between repeated log entries for the same person")
    args = parser.parse_args()

    sys.exit(run(args.camera_index, args.threshold, args.cooldown))
