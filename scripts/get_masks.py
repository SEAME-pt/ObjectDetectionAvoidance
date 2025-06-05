import numpy as np
import cv2
import os

def extract_colored_class_masks(results, output_dir=None, image_name=None, class_colors=None):

    # Default color map for classes (class_id: (R, G, B))
    if class_colors is None:
        class_colors = {
            3: (0, 0, 255),  # 'lane' -> Blue
            2: (255, 0, 0),  # 'car' -> Red
            0: (0, 255, 0),  # 'person' -> Green
            11: (255, 255, 0),  # 'stop sign' -> Yellow
        }

    # Initialize colored mask with original image size
    orig_shape = results[0].orig_img.shape[:2]  # H, W (e.g., 480, 1280)
    colored_mask = np.zeros((*orig_shape, 3), dtype=np.uint8)

    for result in results:
        masks = result.masks
        if masks is not None:
            # Get masks and class IDs
            masks_np = masks.data.cpu().numpy()  # Shape: (N, H, W)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Shape: (N,)

            if masks_np.shape[0] == 0:
                print("No masks detected in this result.")
                continue

            # Process each mask
            for i, (mask, class_id) in enumerate(zip(masks_np, class_ids)):
                if class_id in class_colors:
                    # Convert mask to binary
                    binary_mask = (mask > 0.1).astype(np.uint8)

                    # Resize mask to original image size
                    binary_mask = cv2.resize(binary_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Apply color to corresponding pixels
                    color = class_colors[class_id]
                    mask_indices = binary_mask == 1
                    # Only color uncolored pixels
                    colored_mask[mask_indices] = np.where(
                        colored_mask[mask_indices].sum(axis=1, keepdims=True) == 0,
                        color,
                        colored_mask[mask_indices]
                    )

    # Save colored mask
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        mask_path = os.path.join(output_dir, f"{image_name}_colored_mask.png")
        cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        print(f"Saved colored mask to {mask_path}")

    # Visualize (Jetson-compatible)
    # cv2.imshow("Colored Class Masks", cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)

    if colored_mask.sum() == 0:
        print("No masks found in any results.")

    return colored_mask

# Your main loop (unchanged)
if __name__ == "__main__":
    from ultralytics import YOLO

    # Load model
    model_path = "/home/seame/ObjectDetectionAvoidance/models/yolo-object-lane-unfroze/weights/last.pt"
    model = YOLO(model_path)

    # Process images
    folder_path = "/home/seame/seame_road"
    output_dir = "/home/seame/ObjectDetectionAvoidance/masks_last_conf"
    custom_colors = {
        3: (0, 0, 255),  # 'lane' -> Blue
        12: (255, 0, 0),  # 'car' -> Red
    }

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.jpg') and os.path.isfile(file_path):
            results = model.predict(file_path, conf=0.1, imgsz=320)
            image_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"Processing {image_name}...")
            colored_mask = extract_colored_class_masks(results, output_dir, image_name, class_colors=custom_colors)
    # cv2.destroyAllWindows()