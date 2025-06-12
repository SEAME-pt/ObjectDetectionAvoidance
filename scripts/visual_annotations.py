import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import shutil

def load_annotations(label_path, img_width=320, img_height=320, normalize=True):
    annotations = []
    if not os.path.exists(label_path):
        print(f"Annotation file not found: {label_path}")
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
                polygon = []
                box = None

                # Determine number of points (excluding class_id and optional box)
                remaining = parts[1:]
                if len(remaining) >= 4 and len(remaining) % 2 == 0:  # Polygon only
                    num_points = len(remaining) // 2
                    box_parts = []
                elif len(remaining) >= 8 and (len(remaining) - 4) % 2 == 0:  # Polygon + box
                    num_points = (len(remaining) - 4) // 2
                    box_parts = remaining[num_points * 2:]
                else:
                    print(f"Invalid annotation format in {label_path}: {line.strip()}")
                    continue

                # Parse polygon
                for i in range(num_points):
                    x = float(remaining[i * 2])
                    y = float(remaining[i * 2 + 1])
                    if normalize:
                        x *= img_width
                        y *= img_height
                    x = np.clip(x, 0, img_width)
                    y = np.clip(y, 0, img_height)
                    polygon.append([x, y])

                # Parse box (if present)
                if box_parts:
                    x_center = float(box_parts[0])
                    y_center = float(box_parts[1])
                    width = float(box_parts[2])
                    height = float(box_parts[3])
                    if normalize:
                        x_center *= img_width
                        y_center *= img_height
                        width *= img_width
                        height *= img_height
                    x_left = np.clip(x_center - width / 2, 0, img_width)
                    y_top = np.clip(y_center - height / 2, 0, img_height)
                    x_right = np.clip(x_center + width / 2, 0, img_width)
                    y_bottom = np.clip(y_center + height / 2, 0, img_height)
                    box = [x_left, y_top, x_right, y_bottom]

                annotations.append({'class_id': class_id, 'box': box, 'polygon': polygon})
                print(f"Loaded annotation: class_id={class_id}, polygon_points={len(polygon)}, box={box}")
            except ValueError as e:
                print(f"Error parsing annotation in {label_path}: {line.strip()} ({e})")
                continue
    return annotations

def visualize_annotations(image, annotations):
    # Validate image
    if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image must be a 3-channel RGB numpy array")
        return image
    
    vis_img = image.copy()
    img_height, img_width = image.shape[:2]
    print(f"Visualizing on image size: {img_width}x{img_height}")

    # Define colors (RGB)
    colors = {
        12: (0, 255, 0),  # Green for drivable area
        3: (0, 0, 255),   # Red for lane lines (matches your mask context)
        11: (255, 0, 0)   # Blue for other class (if needed)
    }

    # Draw annotations
    for ann in annotations:
        class_id = ann['class_id']
        box = ann['box']
        polygon = ann['polygon']
        class_name = f"Class {class_id}"
        color = colors.get(class_id, (255, 255, 0))  # Default yellow if class_id not in colors
        print(f"Drawing annotation: class_id={class_id}, color={color}")

        # Draw bounding box 
        if box and class_id != 3 and class_id != 12 :
            x_left, y_top, x_right, y_bottom = map(int, box)
            cv2.rectangle(vis_img, (x_left, y_top), (x_right, y_bottom), color, 2)
            cv2.putText(vis_img, class_name, (x_left, y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw polygon
        if polygon:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=2)
    
    return vis_img


def verify_dataset(image_dir, label_dir,  output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Randomly sample images
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Statistics
    total_boxes = 0
    total_segments = 0

    invalid_annotations = 0
    missing_labels = 0
    
    for img_name in tqdm(sample_files, desc="Verifying annotations"):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        height, width = img.shape[:2]

        annotations = load_annotations(label_path, width, height)
        if not annotations and not os.path.exists(label_path):
            print(f"Missing label file: {label_path}")
            missing_labels += 1
        
        # Check annotations
        for ann in annotations:
            class_id = ann['class_id']
            box = ann['box']
            polygon = ann['polygon']

            # Validate box coordinates
            x_left, y_top, x_right, y_bottom = box
            if not (0 <= x_left < x_right <= width and 0 <= y_top < y_bottom <= height):
                print(f"Invalid box coordinates in {label_path}: {box}")
                invalid_annotations += 1
            
            # Validate polygon
            if polygon and class_id == 12:
                total_segments += 1
                for x, y in polygon:
                    if not (0 <= x <= width and 0 <= y <= height):
                        print(f"Invalid polygon point in {label_path}: ({x}, {y})")
                        invalid_annotations += 1
            
            total_boxes += 1
        
        # Visualize
        vis_img = visualize_annotations(img, annotations)
        output_path = os.path.join(output_dir, f"{img_name}")
        cv2.imwrite(output_path, vis_img)

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Total segments: {total_segments}")
    print(f"Invalid annotations: {invalid_annotations}")
    print(f"Missing label files: {missing_labels}")

def main():
    shutil.rmtree('/home/seame/ObjectDetectionAvoidance/verify', ignore_errors=True)
    image_dir = '../dataset/images/train'  # Adjust this path as needed
    label_dir = '../dataset/labels/train'  # Adjust this path as needed
    output_dir = '/home/seame/ObjectDetectionAvoidance/verify'
    num_samples = 10000
    verify_dataset(image_dir, label_dir, output_dir, num_samples)

if __name__ == '__main__':
    main()