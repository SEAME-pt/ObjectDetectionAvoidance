import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
from pathlib import Path

model_path = "../models/yolo-object-lane-unfroze/weights/best.pt"  
# model_path = "../pretrained_yolo/yolov8n-seg.pt"  # Path to the trained YOLOv8 model

model = YOLO(model_path)

# Define the folder path
image_dir = "/home/seame/frames/frames3"
# model.predict('/home/seame/download.jpeg', save=True, imgsz=320, conf=0.3)
# model.predict('/home/seame/download (1).jpeg', save=True, imgsz=320, conf=0.3)

for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)
    if os.path.isfile(file_path):
        model.predict(file_path, save=True, imgsz=320, conf=0.1)


# specific_class_id = 3  # Replace with the class ID for lane markings

# jpg_files = [
#     f for f in os.listdir(image_dir)
#     if f.startswith('frame_') and f.lower().endswith('.jpg')
# ]

# for jpg in jpg_files:
#     image_path = os.path.join(image_dir, jpg)
#     label_path = os.path.join(image_dir, os.path.splitext(jpg)[0] + '.txt')
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Failed to load image: {image_path}")
#         continue
    
#     # Load annotations from file
#     img_height, img_width = image.shape[:2]
    
#     # Run YOLO inference
#     results = model.predict(source=image, conf=0.1)
    
#     # Extract and process masks for specific class
#     lane_mask = None
#     for result in results:
#         masks = result.masks
#         if masks is not None:
#             # class_ids = result.boxes.cls.cpu().numpy()
#             masks_np = masks.data.cpu().numpy()
#             class_mask_indices = np.where(class_ids == specific_class_id)[0]
#             if len(class_mask_indices) > 0:
#                 specific_masks = masks_np[class_mask_indices]
#                 lane_mask = np.any(specific_masks, axis=0).astype(np.uint8) * 255
#             else:
#                 print(f"No masks found for class ID {specific_class_id} in {image_path}")
#                 lane_mask = np.zeros((img_height, img_width), dtype=np.uint8)