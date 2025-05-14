import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO
import shutil

def generate_synthetic_object_annotations(model_path, img_dir, output_dir, split):
    label_dir = os.path.join(output_dir, 'labels_objects', split)
    os.makedirs(label_dir, exist_ok=True)
    model = YOLO(model_path)
    
    for img_name in tqdm(os.listdir(os.path.join(img_dir, split)), desc=f"Generating object annotations ({split})"):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Failed to load {img_name}")
            continue
        img_path = os.path.join(img_dir, split, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        h, w = img.shape[:2]
        
        # Run YOLO detection
        results = model(img_path, conf=0.6, imgsz=320, verbose=False)
        labels = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x_center, y_center, width, height = box.xywhn[0].tolist()  # Normalized
                # Create dummy polygon (box corners)
                x_left = x_center - width / 2
                x_right = x_center + width / 2
                y_top = y_center - height / 2
                y_bottom = y_center + height / 2
                dummy_polygon = [
                    x_left, y_top,
                    x_right, y_top,
                    x_right, y_bottom,
                    x_left, y_bottom
                ]
                for i in range(0, len(dummy_polygon), 2):
                    x, y = dummy_polygon[i], dummy_polygon[i + 1]
                    dummy_polygon[i] = max(0, min(1, x))
                    dummy_polygon[i + 1] = max(0, min(1, y))
                labels.append(f"{class_id} {' '.join(f'{p:.6f}' for p in dummy_polygon)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save annotations
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            if labels:
                f.write('\n'.join(labels))

def convert_lane_masks_to_polygons(output_dir, split, mask_dir):
    label_dir = os.path.join(output_dir, 'labels_lanes', split)
    os.makedirs(label_dir, exist_ok=True)
    for mask_name in tqdm(os.listdir(mask_dir), desc=f"Converting lane masks ({split})"):
        if not mask_name.endswith('.png'):
            print(f"Failed to load {mask_name}")
            continue
        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load {mask_path}")
            continue
        h, w = mask.shape  # 320x320
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found in {mask_path}")
            continue
        labels = []
        for contour in contours:
            # Normalize polygon points
            if len(contour) < 3:
                print(f"No contours found in {mask_path}")
                continue
            polygon = contour.squeeze()
            polygon_norm = []
            for x, y in polygon:
                # polygon_norm.extend([x / w, y / h])
                x_norm = x / w
                y_norm = y / h # Adjust for letterboxing (320x360 content, 140px top padding)
                # Clamp to [0, 1]
                x_norm = max(0, min(1, x_norm))
                y_norm = max(0, min(1, y_norm))
                polygon_norm.extend([x_norm, y_norm])
            # Compute bounding box from polygon
            x_coords, y_coords = polygon[:, 0], polygon[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            labels.append(f"80 {' '.join(f'{p:.6f}' for p in polygon_norm)} {x_center} {y_center} {width} {height}")
        
        # Append to existing label file
        label_path = os.path.join(label_dir, mask_name.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            if labels:
                f.write('\n'.join(labels) + '\n')
            else:
                print(f"No valid polygons found for {mask_name}")

def merge_annotations(output_dir, split):
    lane_label_dir = os.path.join(output_dir, 'labels_lanes', split)
    object_label_dir = os.path.join(output_dir, 'labels_objects', split)
    merged_label_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(merged_label_dir, exist_ok=True)
    
    all_labels = set(os.listdir(object_label_dir)) | set(os.listdir(lane_label_dir))
    
    for label_name in tqdm(all_labels, desc=f"Merging annotations ({split})"):
        merged_labels = []
        
        # Read object labels
        object_path = os.path.join(object_label_dir, label_name)
        if os.path.exists(object_path):
            with open(object_path, 'r') as f:
                merged_labels.extend(f.read().strip().split('\n'))
        
        # Read lane labels
        lane_path = os.path.join(lane_label_dir, label_name)
        if os.path.exists(lane_path):
            with open(lane_path, 'r') as f:
                merged_labels.extend(f.read().strip().split('\n'))
        
        # Save merged labels
        merged_path = os.path.join(merged_label_dir, label_name)
        with open(merged_path, 'w') as f:
            if merged_labels:
                f.write('\n'.join(merged_labels) + '\n')
            else:
                print(f"No annotations for {label_name}")

# Example usage
output_dir = '/home/seame/ObjectDetectionAvoidance/dataset'
img_dir = '/home/seame/ObjectDetectionAvoidance/dataset/images'
mask_dir_train = '/home/seame/ObjectDetectionAvoidance/dataset/masks/train'
mask_dir_val = '/home/seame/ObjectDetectionAvoidance/dataset/masks/val'

shutil.rmtree(os.path.join(output_dir, 'labels'), ignore_errors=True)
# shutil.rmtree(os.path.join(output_dir, 'labels_objects'), ignore_errors=True)
# shutil.rmtree(os.path.join(output_dir, 'labels_lanes'), ignore_errors=True)

generate_synthetic_object_annotations('../pretrained_yolo/yolo11n.pt', img_dir, output_dir, 'train')
convert_lane_masks_to_polygons(output_dir, 'train', mask_dir_train)

generate_synthetic_object_annotations('../pretrained_yolo/yolo11n.pt', img_dir, output_dir, 'val')
convert_lane_masks_to_polygons(output_dir, 'val', mask_dir_val)

merge_annotations(output_dir, 'train')
merge_annotations(output_dir, 'val')

shutil.rmtree(os.path.join(output_dir, 'labels_objects'), ignore_errors=True)
shutil.rmtree(os.path.join(output_dir, 'labels_lanes'), ignore_errors=True)