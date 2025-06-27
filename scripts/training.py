from ultralytics import YOLO
import shutil
import os

def copy_files(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    # Iterate through all files in source directory
    for item in os.listdir(source_dir):
        if item == "old_models":
            continue
        source_path = os.path.join(source_dir, item)
        dest_path = os.path.join(dest_dir, item)
        print(f"Copying {item} from {source_path} to {dest_path}")
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)

source_directory = "../models/"
destination_directory = "../models/old_models/"
# copy_files(source_directory, destination_directory)
# shutil.rmtree("../models/yolo-object-lane/", ignore_errors=True)
# shutil.rmtree("../models/yolo-object-lane-unfroze/", ignore_errors=True)


# model = YOLO("yolov8l-seg.pt")
model = YOLO("../models/bdd100k/weights/best.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=100,
    warmup_epochs=3,  # Warmup epochs
    imgsz=640,  # Increased for better detail
    hsv_h=0.015,
    hsv_s=0.7,  # Stronger HSV for lighting variations
    hsv_v=0.4,
    translate=0.1,  # Moderate translation
    scale=0.3,  # Reduced for segmentation precision
    fliplr=0.5,  # Kept for driving scenarios
    mosaic=0.7,  # Slightly reduced for segmentation
    erasing=0.2,  # Reduced to avoid excessive mask corruption
    batch=16,  # Feasible with 16GB VRAM for imgsz=640
    device=0,
    workers=8,
    # augment=transform,
    auto_augment=None,  # Disable auto-augmentation
    project="../models",
    name="bdd100k2",
    exist_ok=True,
    freeze=10,  # Freeze backbone
    lr0=0.001,
    patience=20,  # Early stopping
    weight_decay=0.0005
)


model = YOLO("../models/bdd100k2/weights/best.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/second_training/data.yaml",
    epochs=100,
    warmup_epochs=3,
    imgsz=320,
    hsv_h=0.015,
    hsv_s=0.7,  # Stronger HSV for lighting variations
    hsv_v=0.4,
    translate=0.1,  # Moderate translation
    scale=0.3,  # Reduced for segmentation precision
    fliplr=0.5,  # Kept for driving scenarios
    mosaic=0.7,  # Slightly reduced for segmentation
    erasing=0.2,  # Reduced to avoid excessive mask corruption
    batch=16,  
    device=0,
    workers=8,
    auto_augment=None,  # Disable auto-augmentation
    project="../models",
    name="second_training2",
    exist_ok=True,
    freeze=None,  # Unfreeze all layers
    lr0=0.002, 
    patience=20,  # Early stopping
    weight_decay=0.0005,
    mixup=0.1,  # Add mixup for small dataset
    copy_paste=0.3  #
)