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


# def process_multi_class_mask(mask_path, class_id_map={1: 12, 2: 3}, orig_width=320, orig_height=320, padded_size=320):
#     # Read the mask
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         print(f"Failed to load mask: {mask_path}")
#         return []

#     # Debug directory for saving intermediate masks
#     debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
#     os.makedirs(debug_dir, exist_ok=True)

#     # Calculate padding and content height
#     content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
#     content_height = int(content_height)  # e.g., 180 pixels
#     pad_y = (padded_size - content_height) / 2  # e.g., 70 pixels top and bottom
#     pad_x = 0

#     # List to store all polygons with remapped class IDs
#     all_polygons = []

#     # Process each class
#     for input_class_id in class_id_map.keys():
#         # Create binary mask for the current class
#         binary_mask = (mask == input_class_id).astype(np.uint8)  # 1 where mask == input_class_id, 0 elsewhere

#         binary_mask = smooth_mask(binary_mask)
#         debug_path = os.path.join(debug_dir, f"{os.path.basename(mask_path).replace('.png', '')}_class{input_class_id}.png")
#         cv2.imwrite(debug_path, binary_mask * 255)
#         polygons = mask_to_polygons(binary_mask)

#         # Process polygons
#         for poly in polygons:
#             # Adjust and scale coordinates
#             adjusted_coords = poly - np.array([pad_x, pad_y])
#             scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
#             flat_coords = scaled_coords.flatten()
#             normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
#             normalized_coords = np.clip(normalized_coords, 0, 1)
#             print(f"Normalized coordinates for class {input_class_id}: {len(normalized_coords)}")
#             # Use remapped class ID
#             output_class_id = class_id_map[input_class_id]
#             all_polygons.append([output_class_id] + normalized_coords.tolist())

#     return all_polygons


# def process_multi_class_mask(mask_path, orig_width=320, orig_height=320, padded_size=320):
#     # Read the mask
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         print(f"Failed to load mask: {mask_path}")
#         return []

#     # Debug directory for saving intermediate masks
#     debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
#     os.makedirs(debug_dir, exist_ok=True)

#     # Calculate padding and content height
#     content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
#     content_height = int(content_height)  # e.g., 320 if orig_width == orig_height
#     pad_y = (padded_size - content_height) / 2  # e.g., 0 if content_height == padded_size
#     pad_x = 0
#     print(f"Content height: {content_height}, pad_x: {pad_x}, pad_y: {pad_y}")

#     # Get unique non-zero class IDs from the mask
#     class_ids = np.unique(mask[mask != 0])
#     print(f"Class IDs found in mask: {class_ids.tolist()}")

#     # List to store all polygons with original class IDs
#     all_polygons = []

#     # Process each class
#     for class_id in class_ids:
#         # Create binary mask for the current class
#         binary_mask = (mask == class_id).astype(np.uint8)  # 1 where mask == class_id, 0 elsewhere
#         pixel_count = np.sum(binary_mask)
#         print(f"Class {class_id} pixels: {pixel_count}")

#         # Smooth the binary mask (assumed defined elsewhere)
#         binary_mask = smooth_mask(binary_mask)

#         # Save debug mask
#         debug_path = os.path.join(debug_dir, f"{os.path.basename(mask_path).replace('.png', '')}_class{class_id}.png")
#         cv2.imwrite(debug_path, binary_mask * 255)
#         print(f"Saved debug mask for class {class_id}: {debug_path}")

#         # Extract polygons (assumed defined elsewhere)
#         polygons = mask_to_polygons(binary_mask)
#         print(f"Class {class_id} polygons: {len(polygons)}")

#         # Process polygons
#         for poly in polygons:
#             # Adjust and scale coordinates
#             adjusted_coords = poly - np.array([pad_x, pad_y])
#             scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
#             flat_coords = scaled_coords.flatten()
#             normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
#             normalized_coords = np.clip(normalized_coords, 0, 1)
#             print(f"Normalized coordinates for class {class_id}: {len(normalized_coords)}")
#             # Use original class ID
#             all_polygons.append([int(class_id)] + normalized_coords.tolist())

#     return all_polygons


# def process_multi_class_mask(mask_path, orig_width=320, orig_height=320, padded_size=320):
#     # Read the mask
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         print(f"Failed to load mask: {mask_path}")
#         return []

#     # Validate mask size
#     if mask.shape != (padded_size, padded_size):
#         print(f"Warning: Mask size {mask.shape} does not match expected ({padded_size}, {padded_size})")

#     # Debug directory for saving intermediate masks
#     debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
#     os.makedirs(debug_dir, exist_ok=True)

#     # Calculate padding and content height
#     content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
#     content_height = int(content_height)
#     pad_y = (padded_size - content_height) / 2
#     pad_x = 0
#     print(f"Content height: {content_height}, pad_x: {pad_x}, pad_y: {pad_y}")

#     # Get unique non-zero class IDs from the mask
#     class_ids = np.unique(mask[mask != 0])
#     if len(class_ids) == 0:
#         print(f"No non-zero class IDs found in mask: {mask_path}")
#         return []
#     print(f"Class IDs found in mask: {class_ids.tolist()}")

#     # List to store all polygons with original class IDs
#     all_polygons = []

#     # Process each class
#     for class_id in class_ids:
#         # Create binary mask for the current class
#         binary_mask = (mask == class_id).astype(np.uint8)
#         pixel_count = np.sum(binary_mask)
#         print(f"Class {class_id} pixels: {pixel_count}")

#         # Skip if no pixels for this class (safety check)
#         if pixel_count == 0:
#             print(f"Skipping class {class_id} due to zero pixels")
#             continue

#         # Smooth the binary mask
#         binary_mask = smooth_mask(binary_mask)

#         # Save debug mask
#         debug_path = os.path.join(debug_dir, f"{os.path.basename(mask_path).replace('.png', '')}_class{class_id}.png")
#         cv2.imwrite(debug_path, binary_mask * 255)
#         print(f"Saved debug mask for class {class_id}: {debug_path}")

#         # Extract polygons
#         polygons = mask_to_polygons(binary_mask)
#         print(f"Class {class_id} polygons: {len(polygons)}")

#         # Process polygons
#         for poly in polygons:
#             # Adjust and scale coordinates
#             adjusted_coords = poly - np.array([pad_x, pad_y])
#             scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
#             flat_coords = scaled_coords.flatten()
#             normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
#             normalized_coords = np.clip(normalized_coords, 0, 1)
#             print(f"Normalized coordinates for class {class_id}: {len(normalized_coords)}")
#             # Use original class ID
#             all_polygons.append([int(class_id)] + normalized_coords.tolist())

#     print(f"Total polygons generated: {len(all_polygons)}")
#     return all_polygons


def process_multi_class_mask(mask_path, class_id_map={1: 12, 2: 3}, orig_width=320, orig_height=320, padded_size=320):
    # Read the RGB mask
    mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_rgb is None:
        print(f"Failed to load mask: {mask_path}")
        return []

    # Verify mask size
    if mask_rgb.shape[:2] != (padded_size, padded_size):
        print(f"Warning: Mask size {mask_rgb.shape[:2]} does not match expected ({padded_size}, {padded_size})")
    
    # Debug directory
    debug_dir = '/home/seame/ObjectDetectionAvoidance/debug_masks'
    os.makedirs(debug_dir, exist_ok=True)

    # Convert RGB mask to grayscale with class IDs
    mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    green_mask = np.all(mask_rgb == [0, 255, 0], axis=2)
    red_mask = np.all(mask_rgb == [0, 0, 255], axis=2)
    mask[green_mask] = 1  # Drivable area
    mask[red_mask] = 2    # Lane lines

    # Debug pixel counts
    green_count = np.sum(green_mask)
    red_count = np.sum(red_mask)
    print(f"Green pixels (class 1): {green_count}, Red pixels (class 2): {red_count}")

    # Save grayscale mask for debugging
    debug_grayscale_path = os.path.join(debug_dir, f"{os.path.basename(mask_path).replace('.png', '')}_grayscale.png")
    cv2.imwrite(debug_grayscale_path, mask * 127)  # Scale: 0->0, 1->127, 2->254
    print(f"Saved grayscale mask: {debug_grayscale_path}")

    # Calculate padding
    content_height = padded_size * (orig_height / orig_width) * (padded_size / (padded_size * (orig_height / orig_width)))
    content_height = int(content_height)  # Should be 320
    pad_y = (padded_size - content_height) / 2  # Should be 0
    pad_x = 0
    print(f"Content height: {content_height}, pad_x: {pad_x}, pad_y: {pad_y}")

    # List to store all polygons
    all_polygons = []

    # Process each class
    for input_class_id in class_id_map.keys():
        # Create binary mask for the current class
        binary_mask = (mask == input_class_id).astype(np.uint8)
        pixel_count = np.sum(binary_mask)
        print(f"Class {input_class_id} pixels: {pixel_count}")

        # Save binary mask for debugging
        debug_path = os.path.join(debug_dir, f"{os.path.basename(mask_path).replace('.png', '')}_class{input_class_id}.png")
        cv2.imwrite(debug_path, binary_mask * 255)
        print(f"Saved debug mask for class {input_class_id}: {debug_path}")

        # Apply smoothing
        binary_mask = smooth_mask(binary_mask)

        # Extract polygons
        polygons = mask_to_polygons(binary_mask)
        print(f"Class {input_class_id} polygons: {len(polygons)}")

        # Process polygons
        for poly in polygons:
            # Adjust and scale coordinates
            adjusted_coords = poly - np.array([pad_x, pad_y])
            scaled_coords = adjusted_coords * (np.array([orig_width, orig_height]) / np.array([padded_size, content_height]))
            flat_coords = scaled_coords.flatten()
            normalized_coords = flat_coords / np.array([orig_width, orig_height] * (flat_coords.size // 2))
            normalized_coords = np.clip(normalized_coords, 0, 1)
            
            # Use remapped class ID
            output_class_id = class_id_map[input_class_id]
            all_polygons.append([output_class_id] + normalized_coords.tolist())

    return all_polygons


def process_image(mask_path=None):
    objects_info = []
    if mask_path and os.path.exists(mask_path):
        lane_polygons = process_multi_class_mask(mask_path)
        objects_info.extend(lane_polygons)

    return objects_info

def write_yolo_annotations(output_path, image_name, objects_info):
    base_name = image_name.rsplit('.', 1)[0]
    annotation_file_path = os.path.join(output_path, f"{base_name}.txt")
    
    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            coords = obj_info[1:]  # Remaining coordinates as floats
            line = f"{obj_info[0]} " + ' '.join(f'{x:.6f}' for x in coords) + '\n'
            file.write(line)

def process_directory(mask_dir, output_dir=None):
    # os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(image_extensions)]

    for image_file in tqdm(image_files, desc="Processing images"):
        mask_file = image_file.rsplit('.', 1)[0] + '.png'
        # print(f"Processing image {image_file} with mask {mask_file}")
        mask_path = os.path.join(mask_dir, image_file) if mask_dir and os.path.exists(mask_dir) else None
        # print(f"Processing mask {mask_path}")
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
    output_dir = '../best/label/'
    img_dir = '/home/seame/frames/frames3'
    mask_dir = '../best'
    label_dir = './filtered/labels'
    shutil.rmtree('../debug_masks', ignore_errors=True)
    # mask_dir_val = '/home/seame/ObjectDetectionAvoidance/masks'

    os.makedirs(os.path.join(output_dir), exist_ok=True)
    # generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, '')
    # generate_synthetic_object_annotations('../pretrained_yolo/yolo11n-seg.pt', img_dir, output_dir, 'val')

    process_directory(mask_dir, output_dir)
    # process_directory(mask_dir2, output_dir2)
    # merge_annotations(output_dir, None, label_dir, '../filtered/other/labels' , '')


if __name__ == "__main__":
    main()