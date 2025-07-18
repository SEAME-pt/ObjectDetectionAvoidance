import os
import shutil
import sys
import random
from collections import defaultdict

def separate_files_by_prefix(prefix, source_dir="../split_dataset/val/images", output_dir="separated"):
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(source_dir)
    image_extensions = [".jpg", ".jpeg", ".png"]
    annotation_extensions = [".txt", ".json", ".xml"]  # extend as needed

    matched_basenames = set()

    # Find all base names that start with the given prefix
    for f in files:
        if f.startswith(prefix):
            base, ext = os.path.splitext(f)
            if ext.lower() in image_extensions + annotation_extensions:
                matched_basenames.add(base)

    print(f"Found {len(matched_basenames)} matching file(s) with prefix '{prefix}'.")

    for base in matched_basenames:
        for ext in image_extensions + annotation_extensions:
            full_name = base + ext
            src = os.path.join(source_dir, full_name)
            dst = os.path.join(output_dir, full_name)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Copied: {src} -> {dst}")
                
                
                

def get_class_ids(label_path):
    with open(label_path, "r") as f:
        return {line.split()[0] for line in f if line.strip()}

def count_class_occurrences(label_dir):
    class_counts = defaultdict(int)
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            class_ids = get_class_ids(os.path.join(label_dir, fname))
            for cid in class_ids:
                class_counts[cid] += 1
    return class_counts

def is_valid_split(selected_files, label_dir, min_per_class):
    class_counts = defaultdict(int)
    for fname in selected_files:
        label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(label_path):
            continue
        class_ids = get_class_ids(label_path)
        for cid in class_ids:
            class_counts[cid] += 1
    return all(count >= min_per_class for count in class_counts.values())

def split_dataset(image_dir, label_dir, output_dir, val_ratio=0.2, min_per_class=10, max_attempts=100):
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    for attempt in range(max_attempts):
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        if is_valid_split(train_files, label_dir, min_per_class) and is_valid_split(val_files, label_dir, min_per_class):
            print(f"✅ Valid split found after {attempt + 1} attempts.")
            break
    else:
        print("❌ Could not find a valid split that satisfies class balance.")
        return

    def move_files(files, subset):
        for img in files:
            base = os.path.splitext(img)[0]
            label_file = base + ".txt"
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, subset, "images", img))
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, subset, "labels", label_file))

    move_files(train_files, "train")
    move_files(val_files, "val")
    print(f"✅ Moved {len(train_files)} to train, {len(val_files)} to val.")



if __name__ == "__main__":
    # prefix = "frame_"
    # separate_files_by_prefix(prefix)
    split_dataset(
        image_dir="../separated/train/images",
        label_dir="../separated/train/labels",
        output_dir="../separated/val",
        val_ratio=0.2,
        min_per_class=5
    )
   

