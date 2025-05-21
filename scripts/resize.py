import cv2
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

def resize_image_with_letterbox(img_path, output_path, target_size=(320, 320)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return False
    
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)  # Scale to fit within 320x320
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with padding
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Save
    cv2.imwrite(output_path, canvas)
    return True

def resize_images_in_directory(input_dir, output_dir, target_size=(320, 320)):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(input_dir), desc="Resizing images"):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        # letterbox_resize_mask(img_path, target_size)
        resize_image_with_letterbox(img_path, output_path, target_size)


def resize_images(input_dir, output_dir, target_size=320):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            try:
                img_path = os.path.join(input_dir, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to load {img_path}")
                    continue
                
                # Resize to 320x320 (scale factor 0.5)
                resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                
                # Save to output directory
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized)
                print(f"Resized and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Skipped: {filename} (unsupported format)")

if __name__ == "__main__":
    # input_directory = "../dataset/images/train"  # Path to 640x640 images
    # output_directory = "../dataset/images/train_resized"  # Path for 320x320 images
    # resize_images(input_directory, output_directory)
    input_directory = "./best"  # Path to 640x640 images
    output_directory = "./img_resize"  # Path for 320x320 images
    resize_images(input_directory, output_directory)
    # input_directory = "../dataset/masks/train"  # Path to 640x640 images
    # output_directory = "../dataset/masks/train_resized"  # Path for 320x320 images
    # resize_images(input_directory, output_directory)
    # input_directory = "../dataset/masks/val"  # Path to 640x640 images
    # output_directory = "../dataset/masks/val_resized"  # Path for 320x320 images
    # resize_images(input_directory, output_directory)

# # Resize train and val images
# input_train_dir = '/home/seame/ObjectDetectionAvoidance/seame'
# # input_val_dir = '/home/seame/ObjectDetectionAvoidance/dataset/images/val'
# output_train_dir = '/home/seame/ObjectDetectionAvoidance/seame'
# # output_val_dir = '/home/seame/ObjectDetectionAvoidance/dataset/images_resized/val'

# resize_images_in_directory(input_train_dir, output_train_dir)
# # resize_images_in_directory(input_val_dir, output_val_dir)

# input_train_dir = '/home/seame/ObjectDetectionAvoidance/seame_mask'
# # input_val_dir = '/home/seame/ObjectDetectionAvoidance/dataset/masks/val'
# output_train_dir = '/home/seame/ObjectDetectionAvoidance/seame_mask'
# # output_val_dir = '/home/seame/ObjectDetectionAvoidance/dataset/mask_resized/val'

# resize_images_in_directory(input_train_dir, output_train_dir)
# # resize_images_in_directory(input_val_dir, output_val_dir)
