import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil

# Set paths
model_path = "../runs/train/yolo-seg-lane1-unfrozen/weights/last.pt"  
images_dir = "../dataset/images/val"  # Folder with images to run inference on

shutil.rmtree('/home/seame/ObjectDetectionAvoidance/dataset/results', ignore_errors=True)
output_dir = "/home/seame/ObjectDetectionAvoidance/dataset/results"  # Where to save annotated images
os.makedirs(output_dir, exist_ok=True)

# Load the trained YOLOv8 segmentation model
model = YOLO(model_path)

# Define class names (auto-loaded from model)
class_names = model.names

def visualize_result(frame, result):
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame")

    is_cropped = frame.shape[0] == 360  # 640x360 content region
    frame_height, frame_width = frame.shape[:2]
    y_offset = 140 if is_cropped else 0  # No offset if cropped
    content_height = 360 if is_cropped else 640
    
    # Extract boxes, scores, classes
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    imgsz = result.orig_img.shape[:2] if hasattr(result, 'orig_img') else (640, 640)
    scale_factor = frame_width / imgsz[1]

    # Create a copy of the frame
    vis_frame = frame.copy()

    if masks is not None and len(masks) > 0:
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            if (cls != 80):
                continue
            # Resize mask to frame size
            target_size = (frame_width, frame_height)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 0.5).astype(np.uint8)  # Binarize
            colored_mask = np.stack([mask_resized] * 3, axis=-1) * np.array([255, 0, 0], dtype=np.uint8)  # Blue for lanes
            # Blend mask with image
            vis_frame = np.where(colored_mask, cv2.addWeighted(vis_frame, 0.5, colored_mask, 0.5, 0), vis_frame)

    for box, score, cls in zip(boxes, scores, classes):
        if cls == 80 or score < 0.6:  # Skip boxes for lanes
            continue
        x1, y1, x2, y2 = box.astype(float) * scale_factor
        if is_cropped:
            y1, y2 = y1 - 140, y2 - 140
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
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

        results = model(img, conf=0.5, imgsz=320, verbose=False)  # Run inference, get first result (single image)
        annotated_img = visualize_result(img.copy(), results[0])

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, annotated_img)

if __name__ == "__main__":
    run_inference_and_save()
