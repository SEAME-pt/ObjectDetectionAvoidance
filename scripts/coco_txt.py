import os
import json
import numpy as np
import argparse

def coco_to_seg_txt(coco_json_path, output_txt_dir, target_class_ids=None):
    if not os.path.isfile(coco_json_path):
        print(f"Error: {coco_json_path} is not a valid file")
        return
    if not os.path.isdir(output_txt_dir):
        os.makedirs(output_txt_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Map image IDs to image info
    images = {img['id']: img for img in coco_data['images']}
    print(f"Found {len(images)} images")

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        if target_class_ids and ann['category_id'] not in target_class_ids:
            continue
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image
    processed_images = 0
    total_txts = 0

    for image_id, img_info in images.items():
        filename = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        base_name = os.path.splitext(filename)[0]

        # Validate size (expecting 320x320 from context)
        if width != 320 or height != 320:
            print(f"Warning: Image {filename} size ({width}x{height}) differs from expected (320x320)")

        # Get annotations
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            print(f"No annotations for {filename}")
            continue

        # Collect segmentation lines
        seg_lines = []
        class_ids_used = set()
        for ann in annotations:
            cat_id = ann['category_id']
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                print(f"Invalid bbox for annotation ID {ann['id']} in {filename}")
                continue

            # Extract bbox coordinates
            x_min, y_min, box_width, box_height = bbox

            # Convert to polygon vertices (top-left, top-right, bottom-right, bottom-left)
            x1, y1 = x_min, y_min
            x2, y2 = x_min + box_width, y_min
            x3, y3 = x_min + box_width, y_min + box_height
            x4, y4 = x_min, y_min + box_height

            # Normalize coordinates
            coords = [x1, y1, x2, y2, x3, y3, x4, y4]
            normalized_coords = []
            for i in range(0, len(coords), 2):
                x = coords[i] / width
                y = coords[i + 1] / height
                normalized_coords.extend([x, y])

            # Clip to [0, 1]
            normalized_coords = np.clip(normalized_coords, 0, 1).tolist()

            # Create segmentation line
            seg_line = [cat_id] + normalized_coords
            seg_lines.append(seg_line)
            class_ids_used.add(cat_id)

        # Save TXT file if there are annotations
        if seg_lines:
            txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                for line in seg_lines:
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    f.write(' '.join(map(str, line)) + '\n')
            print(f"Saved segmentation TXT: {txt_path}")
            print(f"Class IDs in {base_name}: {sorted(list(class_ids_used))}")
            total_txts += 1
        else:
            print(f"Skipped saving empty TXT for {filename}")

        processed_images += 1

    print(f"Processed {processed_images} images, saved {total_txts} TXT files")

coco_json_path = '../coco_pri/train/_annotations.coco.json'
output_dir = '../coco_pri/output/'

coco_to_seg_txt(coco_json_path, output_dir)
