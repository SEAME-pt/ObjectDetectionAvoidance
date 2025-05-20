import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil

# Set paths
model_path = "../models/yolo-lane-aug/weights/last.pt"  
# model_path = "../pretrained_yolo/yolov8n-seg.pt"  # Path to the trained YOLOv8 model
images_dir = "../dataset/images/val"  # Folder with images to run inference on

shutil.rmtree('/home/seame/ObjectDetectionAvoidance/model_results', ignore_errors=True)
output_dir = "/home/seame/ObjectDetectionAvoidance/model_results"  # Where to save annotated images
os.makedirs(output_dir, exist_ok=True)

# Load the trained YOLOv8 segmentation model
model = YOLO(model_path)

# # Define class names (auto-loaded from model)
# class_names = model.names

def visualize_result(frame, result):
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame")

    frame_height, frame_width = frame.shape[:2]
    
    # Extract boxes, scores, classes
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    imgsz = result.orig_img.shape[:2] if hasattr(result, 'orig_img') else (320, 320)
    scale_x = frame_width / imgsz[1]
    scale_y = frame_height / imgsz[0]

    # Create a copy of the frame
    vis_frame = frame.copy()

    if masks is not None and len(masks) > 0:
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            # if (cls != 80):
            #     continue
            # Resize mask to frame size
            target_size = (frame_width, frame_height)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 0.5).astype(np.uint8)  # Binarize
            colored_mask = np.stack([mask_resized] * 3, axis=-1) * np.array([255, 0, 0], dtype=np.uint8)  # Blue for lanes
            # Blend mask with image
            vis_frame = np.where(colored_mask, cv2.addWeighted(vis_frame, 0.5, colored_mask, 0.5, 0), vis_frame)

    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.6:  # Skip boxes for lanes
            continue
        # x1, y1, x2, y2 = box.astype(float) * scale_factor
        x1, y1, x2, y2 = box.astype(float)
        x1, y1 = x1 * scale_x, y1 * scale_y
        x2, y2 = x2 * scale_x, y2 * scale_y
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_width - 1, int(x2)), min(frame_height - 1, int(y2))
        label = f"{f'{cls}'} {score:.2f}"
        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for objects
        cv2.putText(vis_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis_frame

def run_inference_and_save():
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_name}")
            continue

        results = model(img)  # Run inference, get first result (single image)
        annotated_img = visualize_result(img.copy(), results[0])

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, annotated_img)

if __name__ == "__main__":
    run_inference_and_save()

