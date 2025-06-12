import os
import shutil
import random

def get_unique_filename(dest_path):
    base, ext = os.path.splitext(dest_path)
    counter = 1
    new_path = dest_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

def split_dataset(image_dir, val_ratio=0.1):
    mask_paths = []
    image_paths = []

    # Collect all image and mask paths
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                mask_path = image_path.replace('.jpg', '.png')
                if os.path.exists(mask_path):  # Ensure the mask exists
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)

    # Shuffle the images and masks
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    image_paths[:], mask_paths[:] = zip(*combined)

    total_images = len(image_paths)
    val_count = int(total_images * val_ratio)

    # Split into training and validation sets
    val_images = image_paths[:val_count]
    val_masks = mask_paths[:val_count]
    train_images = image_paths[val_count:]
    train_masks = mask_paths[val_count:]

    dest_img_train = 'dataset/images/train/'
    dest_mask_train = 'dataset/masks/train/'
    dest_img_val = 'dataset/images/val/'
    dest_mask_val = 'dataset/masks/val/'

    os.makedirs(dest_img_train, exist_ok=True)
    os.makedirs(dest_mask_train, exist_ok=True)
    os.makedirs(dest_img_val, exist_ok=True)
    os.makedirs(dest_mask_val, exist_ok=True)

    # Move training images and masks
    for img_path, mask_path in zip(train_images, train_masks):
        dest_img_path = os.path.join(dest_img_train, os.path.basename(img_path))
        dest_mask_path = os.path.join(dest_mask_train, os.path.basename(mask_path))

        # Handle filename conflicts
        if os.path.exists(dest_img_path) or os.path.exists(dest_mask_path):
            dest_img_path = get_unique_filename(dest_img_path)
            new_mask_filename = os.path.basename(dest_img_path).replace('.jpg', '.png')
            dest_mask_path = os.path.join(dest_mask_train, new_mask_filename)

        shutil.move(img_path, dest_img_path)
        shutil.move(mask_path, dest_mask_path)

    # Move validation images and masks
    for img_path, mask_path in zip(val_images, val_masks):
        dest_img_path = os.path.join(dest_img_val, os.path.basename(img_path))
        dest_mask_path = os.path.join(dest_mask_val, os.path.basename(mask_path))

        # Handle filename conflicts
        if os.path.exists(dest_img_path) or os.path.exists(dest_mask_path):
            dest_img_path = get_unique_filename(dest_img_path)
            new_mask_filename = os.path.basename(dest_img_path).replace('.jpg', '.png')
            dest_mask_path = os.path.join(dest_mask_val, new_mask_filename)

        shutil.move(img_path, dest_img_path)
        shutil.move(mask_path, dest_mask_path)

# Example usage
image_dir = os.path.join('.', 'dataset10')
split_dataset(image_dir, val_ratio=0.1)  # 10% for validation
