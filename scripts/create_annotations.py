import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO
import shutil
from PIL import Image
import supervision
from supervision.detection.utils import mask_to_polygons
import argparse

def generate_synthetic_object_annotations(model_path, img_dir, output_dir, split, imgsz=320):
    shutil.rmtree(os.path.join(output_dir, 'objects', split), ignore_errors=True)
    label_dir = os.path.join(output_dir, 'objects', split)
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
                    # Handle letterboxing: calculate content area (320x180 for 1280x720 aspect ratio)
                    class_id = class_ids[i]
                    if class_id == 0 or class_id == 2 or class_id == 11 : 
                        # print(f"class {class_id}")
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
                    # labels.append([class_id] + normalized_coords.tolist())
            # for box in result.boxes:
            #     class_id = int(box.cls.item())
            #     x_center, y_center, width, height = box.xywhn[0].tolist()  # Normalized
                        # print(f" {class_id} {normalized_coords}")
                        labels.append(f"{class_id} {' '.join(f'{p:.6f}' for p in normalized_coords)}")
        
        # Save annotations
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            if labels:
                print(f"Saving {len(labels)} annotations for {img_name} to {label_path}")
                f.write('\n'.join(labels))

def smooth_mask(binary_mask):
    blurred = cv2.GaussianBlur(binary_mask.astype(np.float32), (3, 3), sigmaX=0.5)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed

def thin_mask(binary_mask):
    kernel = np.ones((3, 3), np.uint8)
    thinned = cv2.erode(binary_mask, kernel, iterations=1)
    return thinned

def process_mask(mask_path, orig_width=320, orig_height=320, padded_size=320):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return []
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask // 255  # Convert to 0/1 for supervision
    # binary_mask = (mask == 255).astype(np.uint8)
    # print(f" {binary_mask.min()} {binary_mask.max()}")
    # binary_mask = thin_mask(binary_mask)
    binary_mask = smooth_mask(binary_mask)
    polygon = mask_to_polygons(binary_mask)
    content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
    content_height = int(content_height)  # 180 pixels
    pad_y = (padded_size - content_height) / 2  # 70 pixels top and bottom
    pad_x = 0

    debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, os.path.basename(mask_path)), binary_mask * 255)
    
    polygons = []
    for poly in polygon:
        adjusted_coords = poly - np.array([pad_x, pad_y])
        scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
        flat_coords = scaled_coords.flatten()
        normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
        normalized_coords = np.clip(normalized_coords, 0, 1)
        polygons.append([12] + normalized_coords.tolist())

    return polygons

def process_image(mask_path=None):
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
            line = f"{12} " + ' '.join(f'{x:.6f}' for x in coords) + '\n'
            file.write(line)

def process_directory(mask_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(image_extensions)]

    for image_file in tqdm(image_files, desc="Processing images"):
        # mask_file = image_file.rsplit('.', 1)[0] + '.png'
        # print(f"Processing image {image_file} with mask {mask_file}")
        mask_path = os.path.join(mask_dir, image_file) if mask_dir and os.path.exists(mask_dir) else None
        print(f"Processing mask {mask_path}")
        objects_info = process_image(mask_path)
        write_yolo_annotations(output_dir, image_file, objects_info)

def merge_annotations(output_dir, lane_label_dir, da_dir, object_label_dir=None ,split=''):
    if lane_label_dir is None:
        lane_labe_dir = None
    else:
        lane_label_dir = os.path.join(lane_label_dir, split)
    object_label_dir = os.path.join(object_label_dir , split)
    da_dir = os.path.join(da_dir, split)
    merged_label_dir = os.path.join(output_dir, 'merged', split)
    print(f"Merging annotations from {lane_label_dir}, {da_dir} into {merged_label_dir}")
    os.makedirs(merged_label_dir, exist_ok=True)
    
    all_labels =  set(os.listdir(object_label_dir)) | set(os.listdir(lane_label_dir)) | set(os.listdir(da_dir))
    
    for label_name in tqdm(all_labels, desc=f"Merging annotations ({split})"):
        merged_labels = []
        object_path = os.path.join(object_label_dir, label_name)
        if os.path.exists(object_path):
            with open(object_path, 'r') as f:
                merged_labels.extend(f.read().strip().split('\n'))

        if lane_labe_dir:
            lane_path = os.path.join(lane_label_dir, label_name)
            if os.path.exists(lane_path):
                with open(lane_path, 'r') as f:
                    merged_labels.extend(f.read().strip().split('\n'))
        
        da_path = os.path.join(da_dir, label_name)
        if os.path.exists(da_path):
            with open(da_path, 'r') as f:
                merged_labels.extend(f.read().strip().split('\n'))

        merged_path = os.path.join(merged_label_dir, label_name)
        with open(merged_path, 'w') as f:
            if merged_labels:
                f.write('\n'.join(merged_labels) + '\n')
            else:
                print(f"No annotations for {label_name}")

def main():
    output_dir = '../speed/train/objects/objects'
    output_dir1 = '../speed/train/'
    output_dir2 = '../dataset/seame_coco/annotations/test'
    img_dir = '../speed/train/images'
    mask_dir = '../dataset/seam/new_masks/train'
    mask_dir2 = '../dataset/seame_coco/new_masks/test'
    shutil.rmtree('../debug_masks', ignore_errors=True)
    # mask_dir_val = '/home/seame/ObjectDetectionAvoidance/masks'

    os.makedirs(os.path.join(output_dir), exist_ok=True)
    # generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, '')
    # generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, 'val')

    # process_directory(img_dir, mask_dir, output_dir)
    # process_directory(mask_dir2, output_dir2)
    merge_annotations(output_dir1, None, '../speed/train/labels_seg', output_dir, '')
    # merge_annotations(output_dir, '')

    # merge_annotations(output_dir, 'train')

if __name__ == "__main__":
    main()