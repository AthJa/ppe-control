import os
import time
from dotenv import load_dotenv

# Load environment variables (for ROBOFLOW_API_KEY)
load_dotenv()

def main():
    print("========================================")
    print(" YOLO PPE Engine - Step 1: Download")
    print("========================================")
    
    # 1. Ensure Kaggle API is authenticated
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle API authenticated.")
    except Exception as e:
        print("[ERROR] Kaggle authentication failed.")
        print("Please ensure your kaggle.json is placed in ~/.kaggle/kaggle.json or C:\\Users\\<User>\\.kaggle\\kaggle.json")
        print(f"Details: {e}")
        return

    # 2. Ensure Roboflow is authenticated
    rf_key = os.environ.get("ROBOFLOW_API_KEY")
    if not rf_key:
        print("[ERROR] ROBOFLOW_API_KEY not found in .env file.")
        return
    else:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=rf_key)
            print("[OK] Roboflow API authenticated.")
        except Exception as e:
            print("[ERROR] Roboflow authentication failed.")
            print(f"Details: {e}")
            return

    os.makedirs("datasets/raw", exist_ok=True)
    os.chdir("datasets/raw")

    # 3. Download Kaggle Datasets
    print("\n--- Downloading Kaggle Datasets ---")
    
    # Search and download SH17
    print("Searching for 'Safe Human 17'...")
    sh17_results = api.dataset_list(search="Safe Human 17")
    if sh17_results:
        sh17_id = sh17_results[0].ref
        print(f"Found: {sh17_id}. Downloading...")
        api.dataset_download_cli(sh17_id, unzip=True, path="sh17")
    else:
        print("[WARNING] Could not find 'Safe Human 17' on Kaggle.")

    # Search and download COVID Face Mask
    print("Searching for 'Face Mask Dataset COVID-19'...")
    mask_results = api.dataset_list(search="Face Mask Dataset COVID-19")
    if mask_results:
        # Taking the first relevant result
        mask_id = mask_results[0].ref
        print(f"Found: {mask_id}. Downloading...")
        api.dataset_download_cli(mask_id, unzip=True, path="covid_mask")
    else:
        print("[WARNING] Could not find 'Face Mask Dataset COVID-19' on Kaggle.")

    # 4. Download Roboflow Datasets
    print("\n--- Downloading Roboflow Datasets ---")
    
    # CPPE-5 
    print("Downloading CPPE-5...")
    try:
        project_cppe = rf.workspace("andrewmvd").project("cppe-5") # andrewmvd is the original author, often used
        dataset_cppe = project_cppe.version(1).download("yolov8")
        if os.path.exists(dataset_cppe.location):
            os.rename(dataset_cppe.location, "cppe-5")
    except Exception as e:
        print(f"[WARNING] CPPE-5 download via Roboflow failed. Trying alternative workspace 'datasets'...")
        try:
            project_cppe = rf.workspace("datasets").project("cppe-5")
            dataset_cppe = project_cppe.version(1).download("yolov8")
            if os.path.exists(dataset_cppe.location):
                os.rename(dataset_cppe.location, "cppe-5")
        except Exception as e2:
            print(f"[ERROR] Could not download CPPE-5 automatically. You may need to manually provide the snippet.")

    # Head Cover (Ivision2)
    print("Downloading 'head cover' by ivision2...")
    try:
        # Based on user notes: project by ivision2
        project_head = rf.workspace("ivision2").project("head-cover")
        dataset_head = project_head.version(1).download("yolov8")
        if os.path.exists(dataset_head.location):
            os.rename(dataset_head.location, "head_cover")
    except Exception as e:
        print(f"[WARNING] 'head cover' download failed. Trying 'head-cover-...' variations")
        try:
            project_head = rf.workspace("ivision").project("head-cover")
            dataset_head = project_head.version(1).download("yolov8")
            if os.path.exists(dataset_head.location):
                os.rename(dataset_head.location, "head_cover")
        except:
            print("[ERROR] Could not download 'head cover'.")

    print("\n========================================")
    print(" DOWNLOAD PHASE COMPLETE ")
    print("========================================")
    print("ACTION REQUIRED:")
    print("1. Please review the downloaded datasets in the 'datasets/raw' folder.")
    print("2. Navigate to 'datasets/raw/head_cover/train/images' and MANUALLY DELETE any non-medical caps (baseball caps, hard hats, etc.).")
    print("3. Delete their corresponding .txt label files in 'datasets/raw/head_cover/train/labels' to keep it balanced, or let the merge script drop images without labels.")
    print("4. Once pruning is done, notify the AI to proceed to Step 2 (Merge & Train).")

if __name__ == "__main__":
    main()
