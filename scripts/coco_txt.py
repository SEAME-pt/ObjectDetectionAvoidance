import json
import os
import numpy as np
import cv2
import shutil

def generate_filled_masks(coco_json_path, image_dir, output_mask_dir, target_category_id=1):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_mask_dir, exist_ok=True)

    images = {img['id']: img for img in coco_data['images']}
    print(f"Found {len(images)} images")

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if ann['category_id'] != target_category_id:
            continue
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    for image_id, img_info in images.items():
        filename = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)

        mask = np.zeros((height, width), dtype=np.uint8)

        # Get annotations
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            print(f"No lane annotations for {filename}")
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}.png"), mask)
            continue

        # Draw filled polygons
        for ann in annotations:
            seg = ann.get('segmentation', [])

            # Convert to polygon points
            points = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [points], 255)

        mask_path = os.path.join(output_mask_dir, f"{base_name}.png")
        cv2.imwrite(mask_path, mask)



coco_json_path = '/home/seame/new_dataset/train/_annotations.coco.json'
image_dir = '/home/seame/new_dataset/train'
output_mask_dir = '/home/seame/new_dataset/mask/'
output_label_dir = '/home/seame/new_dataset/labels/'

generate_filled_masks(coco_json_path, image_dir, output_mask_dir, target_category_id=1)