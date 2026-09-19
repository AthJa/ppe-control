from ultralytics import YOLO
import os

def main():
    print("========================================")
    print(" YOLO PPE Engine - Step 3: Train")
    print("========================================")
    
    data_yaml_path = "datasets/merged_ppe/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] Could not find {data_yaml_path}. Please run step2_merge.py first.")
        return
        
    print(f"Loading YOLO11n (Nano) model for faster training...")
    # Initialize the YOLO11 nano model (trains much faster than small)
    model = YOLO("yolo11n.pt")

    print("\nStarting Training...")
    # Train the model
    # Adjust batch size and workers depending on your local GPU resources
    results = model.train(
        data=os.path.abspath(data_yaml_path),
        epochs=50,             # Reduced from 100 for faster training
        imgsz=416,             # Reduced from 640. Makes training ~2.5x faster!
        batch=80,              # Increased to 64 to fully utilize the RTX 3050 since memory usage is low
        workers=8,             # Number of CPU threads for data loading
        device=0,              # Target GPU 0
        project="runs/detect", # Where to save the results
        name="ppe_engine",     # Name of the training run
        exist_ok=True,         # Overwrite existing run directory
        patience=20            # Early stopping patience
    )
    
    print("\n[SUCCESS] Training complete!")
    print("The best model weights are saved at: runs/detect/ppe_engine/weights/best.pt")

if __name__ == "__main__":
    main()
