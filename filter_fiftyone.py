import fiftyone as fo
import os

def main():
    dataset_dir = "datasets/raw/head_cover"
    output_dir = "datasets/raw/head_cover_filtered"

    print("========================================")
    print(" YOLO PPE Engine - FiftyOne Filtering")
    print("========================================")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    name = "head-cover-dataset"
    # Delete the dataset if it already exists in the local FiftyOne DB
    if name in fo.list_datasets():
        fo.delete_dataset(name)

    print(f"Loading YOLO dataset from {dataset_dir}...")
    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            name=name,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    print(f"Successfully loaded {len(dataset)} samples.")

    print("\nLaunching FiftyOne App...")
    # Launch the session but run it in the background so we can wait for user input
    session = fo.launch_app(dataset)

    print("\n" + "="*50)
    print("INSTRUCTIONS FOR FILTERING:")
    print("1. In the FiftyOne web UI, identify non-medical caps (baseball caps, hard hats, etc.).")
    print("2. Hover over the images you want to remove and click the checkbox in the top-left corner.")
    print("3. Once you have selected all the unwanted images, click the 'Checkmark' icon at the top of the grid.")
    print("4. Select 'Delete selected samples' from the dropdown menu.")
    print("5. When you are completely finished, return to this terminal window and press ENTER.")
    print("="*50 + "\n")

    input("Press ENTER when you are done filtering in the UI to export the dataset... ")

    print(f"\nExporting the filtered dataset to {output_dir}...")
    try:
        dataset.export(
            export_dir=output_dir,
            dataset_type=fo.types.YOLOv5Dataset
        )
        print(f"[SUCCESS] Filtered dataset saved to {output_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to export dataset: {e}")

if __name__ == "__main__":
    main()
