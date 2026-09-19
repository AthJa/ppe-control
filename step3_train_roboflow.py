from ultralytics import YOLO
import os

def main():
    print("========================================")
    print(" YOLO PPE Engine - Train on Roboflow Dataset")
    print("========================================")
    
    data_yaml_path = "datasets/roboflow_dataset/data.yaml"
    
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] Could not find {data_yaml_path}. Make sure the roboflow zip is extracted correctly.")
        return
        
    print(f"Loading YOLO11n (Nano) model...")
    model = YOLO("yolo11n.pt")

    print("\nStarting Training on Roboflow Dataset...")
    results = model.train(
        data=os.path.abspath(data_yaml_path),
        epochs=50,             
        imgsz=416,             
        batch=80,              
        workers=2, # Reduced to fix the Windows Pagefile limit error (Error 1455)             
        device=0,              
        project="runs/detect", 
        name="ppe_engine_roboflow", # Different run name so it doesn't overwrite your merged run
        exist_ok=True,         
        patience=20            
    )
    
    print("\n[SUCCESS] Training complete!")
    print("The best model weights are saved at: runs/detect/ppe_engine_roboflow/weights/best.pt")

if __name__ == "__main__":
    main()
