import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import shutil

def load_annotations(label_path, img_width, img_height):
    annotations = []

    if not os.path.exists(label_path):
        print(f"Annotation file not found: {label_path}")
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # if len(parts) < 9:  # Minimum: class_id + 4 pairs of polygon coords
            #     print(f"Invalid annotation format (too short) in {label_path}: {line.strip()}")
            #     continue

            try:
                class_id = int(parts[0])

                # Read polygon points directly
                polygon = []
                polygon_points = parts[1:]  # Everything after class_id

                if len(polygon_points) % 2 != 0:
                    print(f"Invalid polygon points (not pairs) in {label_path}: {line.strip()}")
                    continue

                for i in range(0, len(polygon_points), 2):
                    x = float(polygon_points[i]) * img_width
                    y = float(polygon_points[i + 1]) * img_height
                    polygon.append([x, y])

                # Compute bounding box from polygon
                x_coords = [pt[0] for pt in polygon]
                y_coords = [pt[1] for pt in polygon]
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(img_width, max(x_coords))
                y_max = min(img_height, max(y_coords))
                box = [x_min, y_min, x_max, y_max]

                annotations.append({'class_id': class_id, 'box': box, 'polygon': polygon})

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
    # print(f"Visualizing on image size: {img_width}x{img_height}")

    # Define colors (RGB)
    colors = {
        0: (255, 255, 0),  # Yellow for person
        1: (0, 255, 0),    # Green for car
        2: (0, 0, 255),    # Red for stop sign
        3: (255, 0, 0),    # Blue for lane
        4: (255, 165, 0),  # Orange for passadeira
        5: (0, 255, 255),  # Cyan for verde
        6: (255, 255, 0),  # Yellow for amarelo
        7: (0, 0, 0),      # Black for vermelho
    }

    # Define class ID to label name mapping
    class_labels = {
        0: "drivable",
        1: "lane",
        2: "passadeira",
        3: "stop sign",
        4: "50",
        5: "80",
        6: "jetracer",
        7: "gate",
        }


    # Draw annotations
    for ann in annotations:
        class_id = ann['class_id']
        box = ann['box']
        polygon = ann['polygon']
        # Get label name or fallback to class_id as string
        label_name = class_labels.get(class_id, f"Class {class_id}")
        # Combine class_id and label_name for display
        display_text = f"{class_id} {label_name}"
        color = colors.get(class_id, (0, 255, 255))  # Default yellow if class_id not in colors
        # print(f"Drawing annotation: class_id={class_id}, label={display_text}, color={color}")
        text_x, text_y = 5, 15
        if polygon :  # Skip lane and drivable area for polygons
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [points], isClosed=True, color=color, thickness=2)
            px, py = map(int, polygon[0])
            text_x, text_y = px, max(10, py - 10)

    # Compute bounding box from polygon points
        if box : 
            # print(f"Drawing bounding box: {class_id}")
            x_left, y_top, x_right, y_bottom = map(int, box)
            x_left = max(0, min(x_left, img_width - 1))
            x_right = max(0, min(x_right, img_width - 1))
            y_top = max(0, min(y_top, img_height - 1))
            y_bottom = max(0, min(y_bottom, img_height - 1))

            cv2.rectangle(vis_img, (x_left, y_top), (x_right, y_bottom), color, 2)
            text_x, text_y = x_left, max(10, y_top - 10)

        cv2.putText(
            vis_img,
            display_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
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
            missing_labels += 1
        
        # Check annotations
        for ann in annotations:
            class_id = ann['class_id']
            box = ann['box']
            polygon = ann['polygon']

            # Validate box coordinates
            
            # x_left, y_top, x_right, y_bottom = box
            # if not (0 <= x_left < x_right <= width and 0 <= y_top < y_bottom <= height):
            #     print(f"Invalid box coordinates in {label_path}: {box}")
            #     invalid_annotations += 1
            
            # Validate polygon
            if polygon:
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



if __name__ == '__main__':
    shutil.rmtree('/home/seame/ObjectDetectionAvoidance/yolo_models/clutter/verify', ignore_errors=True)
    image_dir = '../../seame_sig/train/'  # Adjust this path as needed
    label_dir = '../../seame_sig/labels/'  # Adjust this path as needed
    output_dir = '/home/seame/ObjectDetectionAvoidance/yolo_models/clutter/verify'
    num_samples = 100000
    # image_dir = '../second_training/val'  # Adjust this path as needed
    # label_dir = '../second_training/valtxt'  # Adjust this path as needed# Adjust this path as needed
    verify_dataset(image_dir, label_dir, output_dir, num_samples)