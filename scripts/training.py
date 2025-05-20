from ultralytics import YOLO

model = YOLO("../pretrained_yolo/yolov8n-seg.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=30,
    imgsz=320,
    hsv_h=0.015,        # hue
    hsv_s=0.3,        # saturation
    hsv_v=0.3,        # Brightness/contrast 
    translate=0.0,    # Disable translation
    scale=0.0,        # Disable scaling
    fliplr=0.0,       # Disable horizontal flip
    mosaic=0.0,       # Disable mosaic
    erasing=0.0,      # Disable random erasing
    auto_augment=None,  # Disable auto-augmentation
    # motion_blur=0.3,  # Motion blur with 20% probability
    # motion_blur_kernel=[3, 7],  # Kernel size range
    # gaussian_blur=0.3,  # Gaussian blur with 20% probability
    # gaussian_blur_sigma=[0.1, 2.0],
    # batch=8,
    device=0,
    workers=4,
    project="../models",
    name="yolo-lane-aug2",
    exist_ok=True,
    freeze=10,  # Freeze backbone
    lr0=0.0005,  # Lower learning rate
    patience=10  # Early stopping
)


model = YOLO("../models/yolo-lane-aug2/weights/best.pt")
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
    name="yolo-lane-unfroze",
    exist_ok=True,
    freeze=0,  # Unfreeze all layers
    lr0=0.0001,  # Very low learning rate
    patience=5  # Early stopping
)