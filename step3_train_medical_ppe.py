"""
YOLO PPE Engine - Train on Medical PPE Dataset
================================================
Dataset: Medical PPE.v1i.yolov11.zip (Roboflow export)
Classes: Coverall, Gloves, Goggles, Mask  (4 classes)
Splits:  ~3352 train / 336 valid / 222 test images
"""

from ultralytics import YOLO
import os
import zipfile
import yaml

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH       = os.path.join(BASE_DIR, "Medical PPE.v1i.yolov11.zip")
EXTRACT_DIR    = os.path.join(BASE_DIR, "datasets", "medical_ppe")
DATA_YAML_PATH = os.path.join(EXTRACT_DIR, "data.yaml")


def extract_dataset():
    """Extract the zip file if the dataset folder doesn't already exist."""
    if os.path.exists(DATA_YAML_PATH):
        print("[INFO] Dataset already extracted – skipping.")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] Zip file not found at:\n  {ZIP_PATH}")
        print("Please place 'Medical PPE.v1i.yolov11.zip' in the project root.")
        raise FileNotFoundError(ZIP_PATH)

    print(f"[INFO] Extracting dataset to {EXTRACT_DIR} ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"[INFO] Extraction complete – {len(os.listdir(EXTRACT_DIR))} top-level items.")


def fix_data_yaml():
    """
    Rewrite data.yaml so that train / val / test paths are absolute.
    The Roboflow export uses relative '../train/images' which breaks
    when YOLO resolves them from a different working directory.
    """
    with open(DATA_YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = os.path.join(EXTRACT_DIR, "train", "images")
    cfg["val"]   = os.path.join(EXTRACT_DIR, "valid", "images")
    cfg["test"]  = os.path.join(EXTRACT_DIR, "test",  "images")

    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("[INFO] data.yaml updated with absolute paths:")
    print(f"  train : {cfg['train']}")
    print(f"  val   : {cfg['val']}")
    print(f"  test  : {cfg['test']}")
    print(f"  nc    : {cfg['nc']}")
    print(f"  names : {cfg['names']}")


def train():
    print("\n========================================")
    print(" YOLO PPE Engine – Train: Medical PPE")
    print("========================================\n")

    # 1. Extract & prepare ─────────────────────────────────────────────
    extract_dataset()
    fix_data_yaml()

    # 2. Load model ────────────────────────────────────────────────────
    model_weights = os.path.join(BASE_DIR, "yolo11n.pt")
    print(f"\n[INFO] Loading YOLO11n (Nano) from {model_weights} ...")
    model = YOLO(model_weights)

    # 3. Train ─────────────────────────────────────────────────────────
    print("\n[INFO] Starting training …\n")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=416,
        batch=80,
        workers=2,              # Keep low to avoid Windows pagefile issues
        device=0,               # GPU 0
        project="runs/detect",
        name="ppe_medical",     # Unique run name for this dataset
        exist_ok=True,
        patience=20,            # Early-stopping patience
    )

    # 4. Done ──────────────────────────────────────────────────────────
    best_weights = os.path.join("runs", "detect", "ppe_medical", "weights", "best.pt")
    print("\n[SUCCESS] Training complete!")
    print(f"Best model weights → {best_weights}")
    return results


if __name__ == "__main__":
    train()
