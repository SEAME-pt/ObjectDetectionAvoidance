import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO
import shutil
from PIL import Image
import supervision as sv
from supervision.detection.utils import mask_to_polygons
import argparse

def generate_synthetic_object_annotations(model_path, img_dir, output_dir, split, imgsz=320):
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
        orig_h, orig_w = img.shape[:2]

        # Run YOLO detection
        results = model(img_path, conf=0.6, imgsz=320, verbose=False)
        labels = []
        for result in results:
            if result.masks is None:
                continue 
            masks = result.masks.data.cpu().numpy()  
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  
            for i, mask in enumerate(masks):
                binary_mask = (mask > 0.6).astype(np.uint8)
                polygons = mask_to_polygons(binary_mask)
                for poly in polygons:
                    # Polygons are in imgsz x imgsz space (320x320)
                    # Handle letterboxing: calculate content area (320x180 for 1280x720 aspect ratio)
                    content_height = int(imgsz * (orig_h / orig_w))  # e.g., 180 for 16:9
                    content_width = imgsz  # 320
                    pad_y = (imgsz - content_height) // 2  # e.g., 70 pixels top/bottom
                    pad_x = 0  # No left/right padding
                    
                    # Adjust for letterboxing: subtract padding offset
                    adjusted_coords = poly - np.array([pad_x, pad_y])
                    
                    # Scale to original image dimensions
                    scaled_coords = adjusted_coords * (np.array([orig_w, orig_h]) / np.array([content_width, content_height]))
                    
                    # Flatten and normalize to [0, 1]
                    flat_coords = scaled_coords.flatten()
                    normalized_coords = flat_coords / np.array([orig_w, orig_h] * (flat_coords.size // 2))
                    
                    # Clip to [0, 1]
                    normalized_coords = np.clip(normalized_coords, 0, 1)
                    
                    # Add class ID and coordinates to annotations
                    class_id = class_ids[i]
                    # labels.append([class_id] + normalized_coords.tolist())
            # for box in result.boxes:
            #     class_id = int(box.cls.item())
            #     x_center, y_center, width, height = box.xywhn[0].tolist()  # Normalized
               
                    labels.append(f"{class_id} {' '.join(f'{p:.6f}' for p in normalized_coords)}")
        
        # Save annotations
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            if labels:
                f.write('\n'.join(labels))


# def process_mask2(mask_dir):
#     for mask_file in tqdm(os.listdir(mask_dir), desc=f"Converting lane masks"):
#         mask_path = os.path.join(mask_dir, mask_file)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         # Image processing
#         _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         objects_info = []

#         contour_image = np.zeros_like(mask)
#         bounding_rect_image = mask.copy()
#         yolo_bbox_image = mask.copy()
        
#         for contour in contours:
#             x, y, width, height = cv2.boundingRect(contour)
#             class_label = 80 
#             x_center, y_center, normalized_width, normalized_height = convert_coordinates_to_yolo(mask.shape[1], mask.shape[0], x, y, width, height)
#             objects_info.append((class_label, x_center, y_center, normalized_width, normalized_height))
            
#             cv2.rectangle(bounding_rect_image, (x, y), (x + width, y + height), 255, thickness=2)

#             cv2.drawContours(contour_image, [contour], 0, 255, thickness=2)

#     return objects_info

# def convert_coordinates_to_yolo(image_width, image_height, x, y, width, height):
#     x_center = (x + width / 2) / image_width
#     y_center = (y + height / 2) / image_height
#     normalized_width = width / image_width
#     normalized_height = height / image_height

#     return x_center, y_center, normalized_width, normalized_height

# objects_info = process_mask(mask_path)

def smooth_mask(binary_mask):
    blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (3, 3), sigmaX=0.5)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed

def thin_mask(binary_mask):
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(binary_mask, kernel, iterations=1)
    return thinned

def process_mask(mask_path, orig_width=1280, orig_height=720, padded_size=320):
    # mask_to_polygons = sv.MaskToPolygons()
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return []
    # _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # binary_mask = binary_mask // 255  # Convert to 0/1 for supervision
    binary_mask = (mask == 255).astype(np.uint8)
    binary_mask = smooth_mask(binary_mask)
    polygon = mask_to_polygons(binary_mask)
    content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
    content_height = int(content_height)  # 180 pixels
    pad_y = (padded_size - content_height) / 2  # 70 pixels top and bottom
    pad_x = 0

    debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, os.path.basename(mask_path)), binary_mask * 255)
    
    # polygon_list = sv.mask_to_polygons(binary_mask)
    polygons = []
    for poly in polygon:
    #     epsilon = 0.05 * cv2.arcLength(poly.astype(np.int32), True)
    #     approx = cv2.approxPolyDP(poly.astype(np.int32), epsilon, True)
    #     approx = approx.reshape(-1, 2)
        # if len(approx) < 3 or len(approx) > 18:
        #     print(f"Invalid polygon in {mask_path}: points={len(approx)}")
        #     continue
        # perimeter = cv2.arcLength(poly.astype(np.int32), True)
        # epsilon = 0.05 * perimeter if perimeter > 200 else 0.02 * perimeter
        # approx = cv2.approxPolyDP(poly.astype(np.int32), epsilon, True)
        # approx = approx.reshape(-1, 2)

        # if len(approx) < 3 or len(approx) > 24:
        #     print(f"Invalid polygon in {mask_path}: points={len(approx)}")
        #     continue

        # x_coords = approx[:, 0]
        # width = x_coords.max() - x_coords.min()
        # if width > max_width_px:
        #     print(f"Skipping wide polygon in {mask_path}: width={width}px")
        #     continue

        # poly_norm = []
        # for x, y in approx:
        #     x_norm = max(0, min(1, x / img_width))
        #     y_norm = max(0, min(1, y / img_height))
        #     poly_norm.extend([x_norm, y_norm])

        adjusted_coords = poly - np.array([pad_x, pad_y])
        scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
        flat_coords = scaled_coords.flatten()
        normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
        normalized_coords = np.clip(normalized_coords, 0, 1)
        # flat_coords = poly.flatten()
        # normalized_coords = flat_coords / np.array([width, height] * (flat_coords.size // 2))
        polygons.append([80] + normalized_coords.tolist())

    return polygons

def process_image(img_path, mask_path=None):
    objects_info = []
    if mask_path and os.path.exists(mask_path):
        lane_polygons = process_mask(mask_path)
        objects_info.extend(lane_polygons)

    return objects_info

def write_yolo_annotations(output_path, image_name, objects_info):
    base_name = image_name.rsplit('.', 1)[0]
    annotation_file_path = os.path.join(output_path, f"{base_name}.txt")
    
    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            coords = obj_info[1:]  # Remaining coordinates as floats
            line = f"{80} " + ' '.join(f'{x:.6f}' for x in coords) + '\n'
            # line = ' '.join(f'{x:.6f}' for x in obj_info) + '\n'
            file.write(line)

def process_directory(image_dir, mask_dir=None, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    for image_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, image_file)
        mask_file = image_file.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_file) if mask_dir and os.path.exists(mask_dir) else None

        objects_info = process_image(img_path, mask_path)
        # if objects_info:
        write_yolo_annotations(output_dir, image_file, objects_info)
        # else:
        #     print(f"No annotations for {image_file}")


# def convert_lane_masks_to_polygons(output_dir, split, mask_dir):
#     label_dir = os.path.join(output_dir, 'labels_lanes', split)
#     os.makedirs(label_dir, exist_ok=True)
#     for mask_name in tqdm(os.listdir(mask_dir), desc=f"Converting lane masks ({split})"):
#         if not mask_name.endswith('.png'):
#             print(f"Failed to load {mask_name}")
#             continue
#         mask_path = os.path.join(mask_dir, mask_name)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Failed to load {mask_path}")
#             continue
#         h, w = mask.shape  # 320x320
        
#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             print(f"No contours found in {mask_path}")
#             continue
#         labels = []
#         for contour in contours:
#             epsilon = 0.02 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#             if len(approx) < 3:
#                 print(f"Invalid contour in {mask_path}, points: {len(approx)}")
#                 continue
#             if len(approx) < 4:
#             # Try coarser simplification for small contours
#                 epsilon = 0.05 * cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, epsilon, True)
#                 if len(approx) < 4:
#                     print(f"Contour too small in {mask_path}, points: {len(approx)}")
#                     continue
#             if len(approx) > 6:
#             # Subsample to 6 points
#                 indices = np.linspace(0, len(approx) - 1, 6, dtype=int)
#                 approx = approx[indices]
#             polygon = approx.squeeze()
#             polygon_norm = []
#             for x, y in polygon:
#                 x_norm = max(0, min(1, x / w))
#                 y_norm = max(0, min(1, y / h))
#                 polygon_norm.extend([x_norm, y_norm])
#             labels.append([80] + polygon_norm)
#             polygon = contour.squeeze()
#             polygon_norm = []
#             for x, y in polygon:
#                 # polygon_norm.extend([x / w, y / h])
#                 x_norm = x / w
#                 y_norm = y / h # Adjust for letterboxing (320x360 content, 140px top padding)
#                 # Clamp to [0, 1]
#                 x_norm = max(0, min(1, x_norm))
#                 y_norm = max(0, min(1, y_norm))
#                 polygon_norm.extend([x_norm, y_norm])
#             # Compute bounding box from polygon
#             # x_coords, y_coords = polygon[:, 0], polygon[:, 1]
#             # x_min, x_max = x_coords.min(), x_coords.max()
#             # y_min, y_max = y_coords.min(), y_coords.max()
#             # x_center = (x_min + x_max) / 2 / w
#             # y_center = (y_min + y_max) / 2 / h
#             # width = (x_max - x_min) / w
#             # height = (y_max - y_min) / h

#             labels.append(f"80 {' '.join(f'{p:.6f}' for p in polygon_norm)} {x_center} {y_center} {width} {height}")
        
#         # Append to existing label file
#         label_path = os.path.join(label_dir, mask_name.replace('.png', '.txt'))
#         with open(label_path, 'w') as f:
#             if labels:
#                 # f.write('\n'.join(labels) + '\n')
#                 f.write('\n'.join(' '.join(map(str, label)) for label in labels) + '\n')
#             else:
#                 print(f"No valid polygons found for {mask_name}")

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

def main():
    output_dir = '/home/seame/ObjectDetectionAvoidance/dataset'
    img_dir = '/home/seame/ObjectDetectionAvoidance/dataset/images'
    mask_dir_train = '/home/seame/ObjectDetectionAvoidance/dataset/masks/train'
    mask_dir_val = '/home/seame/ObjectDetectionAvoidance/dataset/masks/val'
    shutil.rmtree(os.path.join(output_dir, 'labels_objects'), ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, 'labels'), ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, 'labels_lanes'), ignore_errors=True)

    generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, 'train')
    # convert_lane_masks_to_polygons(output_dir, 'train', mask_dir_train)
    generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, 'val')
    # convert_lane_masks_to_polygons(output_dir, 'val', mask_dir_val)

    parser = argparse.ArgumentParser(description="Generate YOLO annotations from masks using supervision.")
    parser.add_argument('--image_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/images/val', help='Directory with images')
    parser.add_argument('--mask_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/masks/val', help='Directory with lane masks')
    parser.add_argument('--output_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/labels_lanes/val', help='Directory for YOLO .txt files')
    os.makedirs('/home/seame/ObjectDetectionAvoidance/dataset/labels_lanes/val', exist_ok=True)
    args = parser.parse_args()

    process_directory(args.image_dir, args.mask_dir, args.output_dir)

    parser = argparse.ArgumentParser(description="Generate YOLO annotations from masks using supervision.")
    parser.add_argument('--image_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/images/train', help='Directory with images')
    parser.add_argument('--mask_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/masks/train', help='Directory with lane masks')
    parser.add_argument('--output_dir', default='/home/seame/ObjectDetectionAvoidance/dataset/labels_lanes/train', help='Directory for YOLO .txt files')
    os.makedirs('/home/seame/ObjectDetectionAvoidance/dataset/labels_lanes/val', exist_ok=True)
    args = parser.parse_args()

    process_directory(args.image_dir, args.mask_dir, args.output_dir)
    merge_annotations(output_dir, 'train')
    merge_annotations(output_dir, 'val')

if __name__ == "__main__":
    main()