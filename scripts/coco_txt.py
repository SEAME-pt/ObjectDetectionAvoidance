import json
import os
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt

def generate_filled_masks(coco_json_path, image_dir, output_mask_dir, category_ids=[1, 2, 3]):
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


# def view_mask(mask_dir, image_dir=None, output_plot_dir='plots', max_images=50):
#     # Create output directory for plots
#     os.makedirs(output_plot_dir, exist_ok=True)
    
#     mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
#     mask_files.sort()
    
#     for i, mask_file in enumerate(mask_files[:max_images]):
#         mask_path = os.path.join(mask_dir, mask_file)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Failed to load mask: {mask_path}")
#             continue

#         # plt.figure(figsize=(10, 5))
#         # plt.subplot(1, 2 if image_dir else 1, 1)
#         # # plt.imshow(mask, cmap='jet' if len(np.unique(mask)) > 2 else 'gray')
#         # plt.title(f'Mask: {mask_file}')
#         # plt.axis('off')

#         if image_dir:
#             # Assume image filename matches mask (without _classX)
#             base_name = mask_file.replace('_class1', '').replace('_class2', '').replace('_class3', '')
#             image_path = os.path.join(image_dir, base_name.replace('.png', '.jpg'))  # Adjust extension if needed
#             if os.path.exists(image_path):
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 # plt.subplot(1, 2, 2)
#                 # plt.imshow(image)
#                 mask_colored = np.zeros_like(image)
#                 if len(np.unique(mask)) <= 2:  # Binary mask
#                     mask_colored[mask == 255] = [255, 0, 0]
#                 else:  # Multi-class mask
#                     colors = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}
#                     for val in np.unique(mask):
#                         if val == 0:
#                             continue
#                         mask_colored[mask == val] = colors.get(val, [255, 255, 0])
#                 # plt.imshow(mask_colored, alpha=0.4)
#                 # plt.title(f'Image with Mask Overlay: {base_name}')
#                 # plt.axis('off')

#         # Save the figure
#         output_path = os.path.join(output_plot_dir, f"{os.path.splitext(mask_file)[0]}_plot.png")
#         cv2.imwrite(output_path, cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
#         # plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         print(f"Saved plot to {output_path}")
#         # plt.close()  # Close the figure to free memory

# def view_mask(mask_dir, image_dir=None, output_plot_dir='plots', max_images=5):
#     # Create output directory for plots
#     os.makedirs(output_plot_dir, exist_ok=True)
    
#     mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
#     mask_files.sort()
    
#     for i, mask_file in enumerate(mask_files[:max_images]):
#         mask_path = os.path.join(mask_dir, mask_file)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             print(f"Failed to load mask: {mask_path}")
#             continue

#         # Create figure
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2 if image_dir else 1, 1)
#         plt.imshow(mask, cmap='jet' if len(np.unique(mask)) > 2 else 'gray')
#         plt.title(f'Mask: {mask_file}')
#         plt.axis('off')

#         if image_dir:
#             # Match image filename to mask (remove _classX)
#             base_name = mask_file.replace('_class1', '').replace('_class2', '').replace('_class3', '')
#             image_path = os.path.join(image_dir, base_name.replace('.png', '.jpg'))
#             if os.path.exists(image_path):
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 plt.subplot(1, 2, 2)
#                 plt.imshow(image)
#                 mask_colored = np.zeros_like(image)
#                 if len(np.unique(mask)) <= 2:  # Binary mask
#                     mask_colored[mask == 255] = [255, 0, 0]  # Red overlay
#                 else:  # Multi-class mask
#                     colors = {
#                         1: [0, 255, 0],  # Green for class_id=12 (drivable area)
#                         2: [255, 0, 0],  # Red for class_id=3 (lane lines)
#                         3: [0, 0, 255]   # Blue for others
#                     }
#                     for val in np.unique(mask):
#                         if val == 0:
#                             continue
#                         mask_colored[mask == val] = colors.get(val, [255, 255, 0])  # Yellow default
#                 plt.imshow(mask_colored, alpha=0.4)
#                 plt.title(f'Image with Mask Overlay: {base_name}')
#                 plt.axis('off')

#         # Convert figure to OpenCV image
#         fig = plt.gcf()
#         fig.canvas.draw()
#         img_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img_rgb = img_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,)
#         img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#         # Save with cv2.imwrite
#         output_path = os.path.join(output_plot_dir, f"{os.path.splitext(mask_file)[0]}_output.png")
#         cv2.imwrite(output_path, img_bgr)
#         print(f"Saved image to: {output_path}")
#         plt.close(fig)  # Close figure to free memory

coco_json_path = '../da_obj/train/_annotations.coco.json'
image_dir = '../da_obj/train/'
output_mask_dir = '../da_obj/masks/'

generate_filled_masks(coco_json_path, image_dir, output_mask_dir)

# view_mask(output_mask_dir, image_dir)