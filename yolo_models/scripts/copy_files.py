from collections import defaultdict
import os
import shutil

def copy_files_with_class_ids(txt_dir, img_dir, output_dir, target_class_ids=[4]):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Get all txt files
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        
        # Read txt file and check for target class IDs
        has_target_class = False
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in target_class_ids and int(parts[0]) != 12:
                    has_target_class = True
                    break
        
        if has_target_class:
            # Copy txt file
            shutil.copy2(txt_path, os.path.join(output_dir, 'labels', txt_file))
            print(f"Copied txt: {txt_file}")
            
            # Copy corresponding image (try common extensions)
            base_name = os.path.splitext(txt_file)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                img_file = base_name + ext
                img_path = os.path.join(img_dir, img_file)
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(output_dir, 'images', img_file))
                    print(f"Copied image: {img_file}")
                    break
            else:
                print(f"No image found for {txt_file}")


def get_class_counts(label_dir):
    counts = defaultdict(int)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[parts[0]] += 1
    return counts

def find_images_with_class(label_dir, class_id):
    matching = []
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip().startswith(class_id + " "):
                    matching.append(label_file.replace('.txt', ''))
                    break
    return matching

def move_image_and_label(base_img_dir, base_lbl_dir, val_img_dir, val_lbl_dir, basename):
    img_extensions = ['.jpg', '.jpeg', '.png']
    for ext in img_extensions:
        src_img = os.path.join(base_img_dir, basename + ext)
        if os.path.exists(src_img):
            shutil.move(src_img, os.path.join(val_img_dir, os.path.basename(src_img)))
            break
    label_path = os.path.join(base_lbl_dir, basename + ".txt")
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(val_lbl_dir, os.path.basename(label_path)))

def move_missing_classes_to_val(
    train_img_dir, train_lbl_dir,
    val_img_dir, val_lbl_dir,
    class_ids_to_check, min_required=10
):
    val_counts = get_class_counts(val_lbl_dir)

    for class_id in class_ids_to_check:
        current_count = val_counts.get(class_id, 0)
        needed = max(0, min_required - current_count)
        if needed == 0:
            print(f"✅ Class {class_id} already has {current_count} in val.")
            continue

        candidates = find_images_with_class(train_lbl_dir, class_id)
        moved = 0
        for base in candidates:
            if moved >= needed:
                break
            move_image_and_label(train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, base)
            moved += 1
        print(f"➡️ Moved {moved} images for class {class_id} to val (needed: {needed}).")



if __name__ == "__main__":
    # txt_dir = '../50!/train/labels'  # Directory with txt files
    # img_dir = '../50!/train/images/'   # Directory with images

    # output_dir = '../split_dataset/'  # Output directory
    # copy_files_with_class_ids(txt_dir, img_dir, output_dir)
    # txt_dir = '../split_dataset/val/labels'  # Directory with txt files
    # img_dir = '../split_dataset/val/images/'   # Directory with images
    move_missing_classes_to_val(
        train_img_dir="../split_dataset/train/images",
        train_lbl_dir="../split_dataset/train/labels",
        val_img_dir="../split_dataset/val/images",
        val_lbl_dir="../split_dataset/val/labels",
        class_ids_to_check=["2", "4", "5"],  # specify the class IDs you want to ensure
        min_required=10
    )
