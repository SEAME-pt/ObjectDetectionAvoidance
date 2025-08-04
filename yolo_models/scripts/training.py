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


model = YOLO("yolov8n-seg.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/yolo_models/split_dataset/data.yaml",
    epochs=150,
    warmup_epochs=5,
    imgsz=320,
    hsv_h=0.4, #hue
    hsv_s=0.7,  # saturation
    hsv_v=0.4, #brightness
    translate=0.3,  # Moderate translation
    scale=0.4, 
    batch=16,  
    device=0,
    workers=8,
    project="../models",
    name="objects",
    exist_ok=True,
    freeze=None,  # Unfreeze all layers
    lr0=0.002, 
    patience=20,  # Early stopping
    weight_decay=0.001,
    fliplr=0,  # horizontal flip
    cls=1.5,              # Emphasize classification loss
    box=7.5,              # Default
    dfl=1.5,
    label_smoothing=0.1,
    mosaic=0.05, 
    erasing=0.5,  
    mixup=0.4,  # Add mixup for small dataset
    copy_paste=0.5, 
    auto_augment=None,  # Disable auto-augmentation
)

model = YOLO("../models/objects/weights/best.pt")  # Load the trained model
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/yolo_models/seame_training/data.yaml",
    epochs=150,
    warmup_epochs=5,
    imgsz=512,
    hsv_h=0.5, #hue
    hsv_s=0.7,  # saturation
    hsv_v=0.4, #brightness
    translate=0.4,  # Moderate translation
    scale=0.5, 
    batch=16,  
    device=0,
    workers=8,
    project="../models",
    name="seame_n",
    exist_ok=True,
    freeze=0,  # Unfreeze all layers
    lr0=0.002, 
    patience=20,  # Early stopping
    weight_decay=0.001,
    fliplr=0,  # horizontal flip
    mosaic=0,
    erasing=0.2, 
    cls=1.5,              # Emphasize classification loss
    box=7.5,              # Default
    dfl=1.5,
    label_smoothing=0.15,
    mixup=0,  
    copy_paste=0, 
    auto_augment=None,  # Disable auto-augmentation
)
