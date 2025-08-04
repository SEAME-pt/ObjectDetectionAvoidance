import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
from pathlib import Path

model_path = "../models/seame_n/weights/best.pt"  

model = YOLO(model_path)
results = model.predict("../clutter/video_seame.avi", save=True, conf=0.2, imgsz=512)

image_dir = "../clutter/signal_lab_test"

for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)
    if os.path.isfile(file_path):
        model.predict(file_path, save=True, imgsz=512, conf=0.2)

os.system('yolo val model="../models/seame_n/weights/best.pt" data="/home/seame/ObjectDetectionAvoidance/yolo_models/seame_training/data.yaml"')

