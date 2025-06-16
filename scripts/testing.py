import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
from pathlib import Path

model_path = "../models/yolo-object-lane-unfroze/weights/best.pt"  

model = YOLO(model_path)

image_dir = "/home/seame/frames/frames2"

for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)
    if os.path.isfile(file_path):
        model.predict(file_path, save=True, imgsz=320, conf=0.1)

