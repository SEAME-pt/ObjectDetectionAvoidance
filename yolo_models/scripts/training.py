from ultralytics import YOLO
import shutil
import os
import torch
import random
from collections import defaultdict


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
# shutil.rmtree("../models/second_training/", ignore_errors=True)
# shutil.rmtree("../models/yolo-object-lane-unfroze/", ignore_errors=True)


# model = YOLO("yolov8l-seg.pt")
# model = YOLO("../models/bdd100k/weights/best.pt")
# results = model.train(
#     data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
#     epochs=200,
#     warmup_epochs=3,  # Warmup epochs
#     imgsz=640,  # Increased for better detail
#     hsv_h=0.015,
#     hsv_s=0.7,  # Stronger HSV for lighting variations
#     hsv_v=0.4,
#     translate=0.1,  # Moderate translation
#     scale=0.3,  # Reduced for segmentation precision
#     fliplr=0.5,  # Kept for driving scenarios
#     mosaic=0.7,  # Slightly reduced for segmentation
#     erasing=0.2,  # Reduced to avoid excessive mask corruption
#     batch=16,  # Feasible with 16GB VRAM for imgsz=640
#     device=0,
#     workers=8,
#     # augment=transform,
#     auto_augment=None,  # Disable auto-augmentation
#     project="../models",
#     name="bdd100k2",
#     exist_ok=True,
#     freeze=10,  # Freeze backbone
#     lr0=0.001,
#     patience=20,  # Early stopping
#     weight_decay=0.0005
# )


# model = YOLO("yolov8s-seg.pt")
# results = model.train(
#     data="/home/seame/ObjectDetectionAvoidance/yolo_models/split_dataset/data.yaml",
#     epochs=150,
#     warmup_epochs=3,
#     imgsz=320,
#     hsv_h=0.2, #hue
#     hsv_s=0.6,  # saturation
#     hsv_v=0.3, #brightness
#     translate=0.2,  # Moderate translation
#     scale=0.3, 
#     batch=16,  
#     device=0,
#     workers=8,
#     project="../models",
#     name="objects_training",
#     exist_ok=True,
#     freeze=None,  # Unfreeze all layers
#     lr0=0.002, 
#     patience=10,  # Early stopping
#     weight_decay=0.001,
#     fliplr=0,  # horizontal flip
#     mosaic=0.05, 
#     erasing=0.5,  
#     mixup=0.5,  # Add mixup for small dataset
#     copy_paste=0.5, 
#     auto_augment=None,  # Disable auto-augmentation
# )

model = YOLO("../models/objects_training/weights/best.pt")  # Load the trained model
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/yolo_models/seame_training/data.yaml",
    epochs=150,
    warmup_epochs=3,
    imgsz=320,
    hsv_h=0.4, #hue
    hsv_s=0.6,  # saturation
    hsv_v=0.3, #brightness
    translate=0.3,  # Moderate translation
    scale=0.3,  # Reduced for segmentation precision
    batch=16,  
    device=0,
    workers=8,
    project="../models",
    name="seame_training52",
    exist_ok=True,
    freeze=None,  # Unfreeze all layers
    lr0=0.002, 
    patience=10,  # Early stopping
    weight_decay=0.001,
    fliplr=0,  # horizontal flip
    mosaic=0,  # Slightly reduced for segmentation
    erasing=0.5, 
    cls_agnostic=True, 
    mixup=0,  
    copy_paste=0, 
    auto_augment=None,  # Disable auto-augmentation
)



# model = YOLO("../pretrained_yolo/yolo11n-seg.pt")  # Use a smaller model for faster training
# results = model.train(
#     data="/home/seame/ObjectDetectionAvoidance/clutter/data.yaml",
#     epochs=1,
#     imgsz=320,
#     batch=16,  
#     device=0,
#     workers=8,
#     auto_augment=None,  # Disable auto-augmentation
#     project="../models",
#     name="test",
#     exist_ok=True,
#     lr0=0.002, 
#     fliplr=0.0,  # horizontal flip
#     mosaic=0.0,  # Slightly reduced for segmentation
#     erasing=0.0,
#     mixup=0,  # Add mixup for small dataset
#     copy_paste=0  # Add copy-paste augmentation
# )