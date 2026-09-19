import os
import shutil
import glob
import yaml
import xml.etree.ElementTree as ET

# Unified Target Classes
# 0: mask, 1: glove, 2: cap
TARGET_MAPPING = {
    # Mask variations
    "mask": 0, "face_mask": 0, "face mask": 0, "with_mask": 0, "medical_mask": 0, "face-mask-medical": 0,
    # Glove variations
    "glove": 1, "gloves": 1, "medical_glove": 1,
    # Cap variations
    "cap": 2, "head cover": 2, "head_cover": 2, "mob_cap": 2, "hair_cover": 2, "helmet": 2
}

RAW_BASE = "datasets/raw"
MERGED_BASE = "datasets/merged_ppe"

def get_class_mapping(yaml_path):
    """Reads data.yaml and returns a dict mapping source_id -> target_id"""
    mapping = {}
    if not os.path.exists(yaml_path):
        return mapping
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    if 'names' in data:
        names = data['names']
        # Handle if names is a dict or list
        if isinstance(names, list):
            names_dict = {i: name for i, name in enumerate(names)}
        else:
            names_dict = names
            
        for src_id, class_name in names_dict.items():
            class_name_lower = str(class_name).lower()
            if class_name_lower in TARGET_MAPPING:
                mapping[int(src_id)] = TARGET_MAPPING[class_name_lower]
                
    return mapping

def convert_voc_to_yolo(xml_path, target_mapping):
    """Parses VOC XML and returns a list of YOLO formatted strings: '<class_id> <x_center> <y_center> <width> <height>'"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    if size is None:
        return []
    
    img_w = float(size.find('width').text)
    img_h = float(size.find('height').text)
    
    if img_w == 0 or img_h == 0:
        return []

    yolo_boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name in target_mapping:
            class_id = target_mapping[name]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            # constrain between 0 and 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
    return yolo_boxes

def merge_head_cover(dataset_name="head_cover"):
    dataset_path = os.path.join(RAW_BASE, dataset_name)
    if not os.path.exists(dataset_path):
        return False
        
    yaml_path = os.path.join(dataset_path, "data.yaml")
    src_to_target = get_class_mapping(yaml_path)
    if not src_to_target:
        print(f"[WARNING] {dataset_name} skipped (no relevant classes or mapping).")
        return False
        
    print(f"Merging {dataset_name} using mapping: {src_to_target}")
    
    merged_count = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")
        if not os.path.exists(img_dir):
            continue
            
        os.makedirs(os.path.join(MERGED_BASE, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MERGED_BASE, split, "labels"), exist_ok=True)
        
        for img_path in glob.glob(os.path.join(img_dir, "*.*")):
            basename = os.path.basename(img_path)
            name, ext = os.path.splitext(basename)
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            
            if os.path.exists(lbl_path):
                valid_boxes = []
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        src_id = int(parts[0])
                        if src_id in src_to_target:
                            target_id = src_to_target[src_id]
                            valid_boxes.append(f"{target_id} " + " ".join(parts[1:]))
                
                # Include image even if valid_boxes is empty (Background image for false positive reduction)
                new_basename = f"{dataset_name}_{basename}"
                new_img_path = os.path.join(MERGED_BASE, split, "images", new_basename)
                new_lbl_path = os.path.join(MERGED_BASE, split, "labels", f"{dataset_name}_{name}.txt")
                shutil.copy(img_path, new_img_path)
                with open(new_lbl_path, 'w') as f:
                    f.write("\n".join(valid_boxes) + "\n" if valid_boxes else "")
                merged_count += 1
                    
    print(f"[SUCCESS] Merged {merged_count} images from {dataset_name}")
    return True

def merge_sh17(dataset_name="sh17"):
    dataset_path = os.path.join(RAW_BASE, dataset_name)
    if not os.path.exists(dataset_path):
        return False
        
    # sh17 has images in `images/` and xmls in `voc_labels/`
    img_dir = os.path.join(dataset_path, "images")
    xml_dir = os.path.join(dataset_path, "voc_labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        print(f"[WARNING] {dataset_name} skipped (missing images/ or voc_labels/).")
        return False
        
    # Read splits
    train_files = set()
    val_files = set()
    try:
        with open(os.path.join(dataset_path, "train_files.txt"), 'r') as f:
            train_files = set([line.strip() for line in f if line.strip()])
        with open(os.path.join(dataset_path, "val_files.txt"), 'r') as f:
            val_files = set([line.strip() for line in f if line.strip()])
    except:
        pass
        
    print(f"Merging {dataset_name} from VOC XML...")
    merged_count = 0
    
    for xml_path in glob.glob(os.path.join(xml_dir, "*.xml")):
        xml_name = os.path.basename(xml_path)
        img_name, _ = os.path.splitext(xml_name)
        
        # Try finding image with common extensions
        img_path = None
        for ext in [".jpeg", ".jpg", ".png"]:
            tmp_path = os.path.join(img_dir, img_name + ext)
            if os.path.exists(tmp_path):
                img_path = tmp_path
                break
                
        if not img_path:
            continue
            
        yolo_boxes = convert_voc_to_yolo(xml_path, TARGET_MAPPING)
        # Determine split
        split = "train"
        basename = os.path.basename(img_path)
        if basename in val_files:
            split = "valid"
            
        os.makedirs(os.path.join(MERGED_BASE, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MERGED_BASE, split, "labels"), exist_ok=True)
        
        new_basename = f"{dataset_name}_{basename}"
        new_img_path = os.path.join(MERGED_BASE, split, "images", new_basename)
        new_lbl_path = os.path.join(MERGED_BASE, split, "labels", f"{dataset_name}_{img_name}.txt")
        
        shutil.copy(img_path, new_img_path)
        with open(new_lbl_path, 'w') as f:
            f.write("\n".join(yolo_boxes) + "\n" if yolo_boxes else "")
        merged_count += 1
            
    print(f"[SUCCESS] Merged {merged_count} images from {dataset_name}")
    return True

def merge_face_mask_detection(dataset_name="face_mask_detection"):
    dataset_path = os.path.join(RAW_BASE, dataset_name)
    if not os.path.exists(dataset_path):
        return False
        
    img_dir = os.path.join(dataset_path, "images")
    xml_dir = os.path.join(dataset_path, "annotations")
    
    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        print(f"[WARNING] {dataset_name} skipped (missing images/ or annotations/).")
        return False
        
    print(f"Merging {dataset_name} from VOC XML...")
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    # Pseudo-random 80/20 split based on filename
    train_split_ratio = 0.8
    train_count = int(len(xml_files) * train_split_ratio)
    
    merged_count = 0
    for i, xml_path in enumerate(xml_files):
        xml_name = os.path.basename(xml_path)
        img_name, _ = os.path.splitext(xml_name)
        
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            tmp = os.path.join(img_dir, img_name + ext)
            if os.path.exists(tmp):
                img_path = tmp
                break
                
        if not img_path:
            continue
            
        yolo_boxes = convert_voc_to_yolo(xml_path, TARGET_MAPPING)
        split = "train" if i < train_count else "valid"
        
        os.makedirs(os.path.join(MERGED_BASE, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MERGED_BASE, split, "labels"), exist_ok=True)
        
        basename = os.path.basename(img_path)
        new_basename = f"{dataset_name}_{basename}"
        new_img_path = os.path.join(MERGED_BASE, split, "images", new_basename)
        new_lbl_path = os.path.join(MERGED_BASE, split, "labels", f"{dataset_name}_{img_name}.txt")
        
        shutil.copy(img_path, new_img_path)
        with open(new_lbl_path, 'w') as f:
            f.write("\n".join(yolo_boxes) + "\n" if yolo_boxes else "")
        merged_count += 1
            
    print(f"[SUCCESS] Merged {merged_count} images from {dataset_name}")
    return True

def create_yaml():
    yaml_content = {
        'path': os.path.abspath(MERGED_BASE).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'mask',
            1: 'glove',
            2: 'cap'
        }
    }
    yaml_path = os.path.join(MERGED_BASE, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created merged data.yaml at {yaml_path}")

def main():
    print("========================================")
    print(" YOLO PPE Engine - Step 2: Merge")
    print("========================================")
    
    os.makedirs(MERGED_BASE, exist_ok=True)
    
    # 1. Merge head_cover (Standard YOLO format)
    merge_head_cover("head_cover")
    
    # 2. Merge sh17 (VOC XML format)
    merge_sh17("sh17")
    
    # 3. Covid mask dataset (Matlab format) skipped
    print("[INFO] Skipping 'covid_mask' dataset because annotations are in Matlab format (.mat).")
    
    # 4. Standard Face Mask Dataset (Kaggle: andrewmvd/face-mask-detection)
    merge_face_mask_detection("face_mask_detection")
    
    create_yaml()
    print("[SUCCESS] Datasets merged successfully.")

if __name__ == "__main__":
    main()
