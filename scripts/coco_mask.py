import json
import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt

def generate_filled_masks(coco_json_path, image_dir, output_mask_dir, category_ids=[2]):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory
    os.makedirs(output_mask_dir, exist_ok=True)

    # Map image IDs to image info
    images = {img['id']: img for img in coco_data['images']}
    print(f"Found {len(images)} images")

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        # if ann['category_id'] not in category_ids:
        #     continue
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image
    for image_id, img_info in images.items():
        filename = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        base_name = os.path.splitext(filename)[0]

        # Create a multi-class mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled polygons for each annotation
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            print(f"No annotations for {filename}")
        else:
            for ann in annotations:
                cat_id = ann['category_id']
                seg = ann.get('segmentation', [])
                points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], cat_id)  # Use category ID as pixel value

        # Save mask
        mask_path = os.path.join(output_mask_dir, f"{base_name}.png")
        cv2.imwrite(mask_path, mask)
        print(f"Saved multi-class mask: {mask_path}")


coco_json_path = '../clutter/lanes/train/_annotations.coco.json'
image_dir = '../clutter/lanes/train/'
output_mask_dir = '../clutter/lanes/masks/'

generate_filled_masks(coco_json_path, image_dir, output_mask_dir)

import matplotlib.pyplot as plt

def plot_mask(mask):
    plt.figure(figsize=(10,10))
    # Use a color map that highlights different integer classes
    plt.imshow(mask, cmap='tab20')  # 'tab20' has distinct colors for up to 20 classes
    plt.colorbar(ticks=np.arange(np.max(mask)+1))
    plt.title("Multi-class Mask Visualization")
    plt.show()

mask='../clutter/lanes/masks/'
# After your mask creation, just call
plot_mask(mask)