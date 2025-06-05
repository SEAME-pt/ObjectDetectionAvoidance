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


model = YOLO("../pretrained_yolo/yolov8n-seg.pt")
# model = YOLO("../old_models/yolo-lane-seame-unfroze/weights/best.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=15,
    imgsz=320,
    hsv_h=0.015,        # hue
    hsv_s=0.3,        # saturation
    hsv_v=0.3,        # Brightness/contrast 
    translate=0.0,    # Disable translation
    scale=0.0,        # Disable scaling
    fliplr=0.0,       # Disable horizontal flip
    mosaic=0.0,       # Disable mosaic
    erasing=0.0,      # Disable random erasing
    # augment=transform,
    auto_augment=None,  # Disable auto-augmentation
    batch=16,
    device=0,
    workers=4,
    project="../models",
    name="test",
    exist_ok=True,
    # freeze=10,  # Freeze backbone
    lr0=0.01,  
    patience=7,  # Early stopping
    weight_decay=0.0005
)


# model = YOLO("../models/yolo-object-lane/weights/best.pt")
# results = model.train(
#     data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
#     epochs=100,
#     imgsz=320,
#     hsv_h=0.3,        # hue
#     hsv_s=0.3,        # saturation
#     hsv_v=0.3,        # Brightness/contrast (±40%)
#     translate=0.0,    # Disable translation
#     scale=0.0,        # Disable scaling
#     fliplr=0.0,       # Disable horizontal flip
#     mosaic=0.0,       # Disable mosaic
#     erasing=0.0,      # Disable random erasing
#     auto_augment=None,  # Disable auto-augmentation
#     batch=16,
#     device=0,
#     workers=4,
#     project="../models",
#     name="yolo-object-lane-unfroze",
#     exist_ok=True,
#     freeze=0,  # Unfreeze all layers
#     lr0=0.001, 
#     patience=50,  # Early stopping
#     weight_decay=0.0005
# )