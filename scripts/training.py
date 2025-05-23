from ultralytics import YOLO
import shutil
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# def copy_files(source_dir, dest_dir):
#     os.makedirs(dest_dir, exist_ok=True)
#     # Iterate through all files in source directory
#     for item in os.listdir(source_dir):
#         source_path = os.path.join(source_dir, item)
#         dest_path = os.path.join(dest_dir, item)
#         shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
#     shutil.rmtree(source_dir, ignore_errors=True)
#     os.makedirs(source_dir, exist_ok=True)

# source_directory = "../models/"
# destination_directory = "../old_models/"
# copy_files(source_directory, destination_directory)

# transform = A.Compose([
#     A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),  # Low-intensity noise, applied to 30% of images
#     A.GaussianBlur(blur_limit=(3, 5), p=0.3),    # Subtle blur, applied to 30% of images
# ], additional_targets={'mask': 'mask'}) # Ensure masks are preserved

# model = YOLO("../pretrained_yolo/yolov8n-seg.pt")
# model = YOLO("../old_models/yolo-lane_trained/weights/best.pt")
# results = model.train(
#     data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
#     epochs=20,
#     imgsz=320,
#     hsv_h=0.015,        # hue
#     hsv_s=0.3,        # saturation
#     hsv_v=0.3,        # Brightness/contrast 
#     translate=0.0,    # Disable translation
#     scale=0.0,        # Disable scaling
#     fliplr=0.0,       # Disable horizontal flip
#     mosaic=0.0,       # Disable mosaic
#     erasing=0.0,      # Disable random erasing
#     # augment=transform,
#     auto_augment=None,  # Disable auto-augmentation
#     batch=8,
#     device=0,
#     workers=4,
#     project="../models",
#     name="yolo-lane_seame_second",
#     exist_ok=True,
#     freeze=10,  # Freeze backbone
#     lr0=0.0001,  # Lower learning rate
#     patience=10  # Early stopping
# )


model = YOLO("../models/yolo-lane_seame/weights/best.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=10,
    imgsz=320,
    hsv_h=0.3,        # hue
    hsv_s=0.3,        # saturation
    hsv_v=0.3,        # Brightness/contrast (±40%)
    translate=0.0,    # Disable translation
    scale=0.0,        # Disable scaling
    fliplr=0.0,       # Disable horizontal flip
    mosaic=0.0,       # Disable mosaic
    erasing=0.0,      # Disable random erasing
    auto_augment=None,  # Disable auto-augmentation
    batch=8,
    device=0,
    workers=4,
    project="../models",
    name="yolo-lane-seame-unfroze",
    exist_ok=True,
    freeze=0,  # Unfreeze all layers
    lr0=0.00005,  # Very low learning rate
    patience=5  # Early stopping
)