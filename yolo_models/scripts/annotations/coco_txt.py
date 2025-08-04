import os
import json
import numpy as np
import argparse

def coco_to_txt(coco_json_path, output_txt_dir):

    if not os.path.isfile(coco_json_path):
        print(f"Error: {coco_json_path} is not a valid file")
        return
    if not os.path.isdir(output_txt_dir):
        os.makedirs(output_txt_dir, exist_ok=True)

    # Load COCO JSON
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading {coco_json_path}: {str(e)}")
        return

    # Map image IDs to image info
    images = {img['id']: img for img in coco_data['images']}
    print(f"Found {len(images)} images in COCO JSON")

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    print(f"Found annotations for {len(annotations_by_image)} images")

    # Process each image
    processed_images = 0
    total_txts = 0

    for image_id, img_info in images.items():
        try:
            filename = img_info['file_name']
            width, height = img_info['width'], img_info['height']
            base_name = os.path.splitext(filename)[0]

            # Validate image size
            if width != height or width not in [320, 416]:
                print(f"Warning: Image {filename} size ({width}x{height}) is not 320x320 or 416x416")

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
           
                if 'segmentation' in ann and ann['segmentation']:
                    # Existing polygon processing (same as your code)
                    polygon = ann['segmentation'][0]
                    if len(polygon) < 4 or len(polygon) % 2 != 0:
                        print(f"Invalid polygon for annotation ID {ann['id']} in {filename}: {polygon}")
                        continue

                    normalized_coords = []
                    for i in range(0, len(polygon), 2):
                        x = polygon[i] / width
                        y = polygon[i + 1] / height
                        normalized_coords.extend([np.clip(x, 0, 1), np.clip(y, 0, 1)])

                    seg_line = [cat_id] + normalized_coords

                elif 'bbox' in ann and ann['bbox']:
                    # bbox format in COCO is [x_min, y_min, width, height]
                    x_min, y_min, w_box, h_box = ann['bbox']
                    x_max = x_min + w_box
                    y_max = y_min + h_box

                    # normalize and create polygon as box corners (clockwise)
                    normalized_coords = [
                        np.clip(x_min / width, 0, 1), np.clip(y_min / height, 0, 1),
                        np.clip(x_max / width, 0, 1), np.clip(y_min / height, 0, 1),
                        np.clip(x_max / width, 0, 1), np.clip(y_max / height, 0, 1),
                        np.clip(x_min / width, 0, 1), np.clip(y_max / height, 0, 1)
                    ]
                    seg_line = [cat_id] + normalized_coords

                seg_lines.append(seg_line)
                class_ids_used.add(cat_id)
            # Save TXT file if there are annotations
            if seg_lines:
                txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
                with open(txt_path, 'w') as f:
                    for line in seg_lines:
                        # Format: class_id x1 y1 x2 y2 ... xn yn
                        f.write(' '.join(map(str, line)) + '\n')
                print(f"Saved segmentation TXT: {txt_path}")
                print(f"Class IDs in {base_name}: {sorted(list(class_ids_used))}")
                total_txts += 1
            else:
                print(f"Skipped saving empty TXT for {filename}")

            processed_images += 1

        except Exception as e:
            print(f"Error processing image ID {image_id} ({filename}): {str(e)}")
            continue

    print(f"Processed {processed_images} images, saved {total_txts} TXT files")


coco_json_path = '../../8080/train'
output_dir = '../../8080/labels'

for file in os.listdir(coco_json_path):
    if file.endswith('.json'):
        coco_json_path = os.path.join(coco_json_path, file)
        print(f"Processing COCO JSON: {coco_json_path}")
        
        coco_to_txt(coco_json_path, output_dir)