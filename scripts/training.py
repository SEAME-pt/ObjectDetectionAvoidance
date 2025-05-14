from ultralytics import YOLO

model = YOLO("../pretrained_yolo/yolov8n-seg.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=30,
    imgsz=320,
    batch=8,
    device=0,
    workers=4,
    project="../models",
    name="yolo-lane",
    exist_ok=True,
    freeze=10,  # Freeze backbone
    lr0=0.0005,  # Lower learning rate
    patience=10  # Early stopping
)


model = YOLO("runs/train/yolo-seg-lane1/weights/best.pt")
results = model.train(
    data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml",
    epochs=10,
    imgsz=320,
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