import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import argparse
import shutil

def load_annotations(label_path, img_width, img_height, content_height=360, top_padding=140):
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if len(parts) > 5:  # Polygon + box
                num_points = (len(parts) - 5) // 2
                polygon = []
                for i in range(num_points):
                    x = float(parts[1 + i * 2]) * img_width
                    y = float(parts[2 + i * 2]) * content_height + top_padding 
                    polygon.append([x, y])
                x_center = float(parts[1 + num_points * 2]) * img_width
                y_center = float(parts[2 + num_points * 2]) * img_height
                width = float(parts[3 + num_points * 2]) * img_width
                height = float(parts[4 + num_points * 2]) * img_height
                box = [x_center - width / 2, y_center - height / 2,
                       x_center + width / 2, y_center + height / 2]
            else:
                print(f"Invalid annotation format in {label_path}: {line.strip()}")
                continue
            annotations.append({'class_id': class_id, 'box': box, 'polygon': polygon})
    return annotations

def visualize_annotations(image, annotations):
    vis_img = image.copy()
    
    # Draw annotations
    for ann in annotations:
        class_id = ann['class_id']
        box = ann['box']
        polygon = ann['polygon']
        class_name = f"{class_id}"
        if (class_id == 80):
            color = (0, 255, 0) 
        else: 
            color = (255, 0, 0) 
        
        # Draw bounding box
        if class_id != 80:
            x_left, y_top, x_right, y_bottom = map(int, box)
            cv2.rectangle(vis_img, (x_left, y_top), (x_right, y_bottom), color, 2)
            cv2.putText(vis_img, class_name, (x_left, y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw polygon
        if class_id == 80 and polygon:
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
            if polygon:
                total_segments += 1
                for x, y in polygon:
                    if not (0 <= x <= width and 0 <= y <= height):
                        print(f"Invalid polygon point in {label_path}: ({x}, {y})")
                        invalid_annotations += 1
            
            total_boxes += 1
        
        # Visualize
        vis_img = visualize_annotations(img, annotations)
        output_path = os.path.join(output_dir, f"vis_{img_name}")
        cv2.imwrite(output_path, vis_img)

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Total segments: {total_segments}")
    print(f"Invalid annotations: {invalid_annotations}")
    print(f"Missing label files: {missing_labels}")

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO dataset annotations.")
    shutil.rmtree('/home/seame/ObjectDetectionAvoidance/dataset/verify')
    parser.add_argument('--image_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/images/train',
                        help='Directory with images')
    parser.add_argument('--label_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/labels/train',
                        help='Directory with YOLO annotations')
    parser.add_argument('--output_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/verify',
                        help='Directory to save visualized images')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of images to visualize')
    args = parser.parse_args()

    verify_dataset(args.image_dir, args.label_dir, args.output_dir, args.num_samples)

if __name__ == '__main__':
    main()