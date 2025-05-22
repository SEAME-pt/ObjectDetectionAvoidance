import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil

model_path = "../models/yolo-lane-unfroze/weights/best.pt"  
# model_path = "../pretrained_yolo/yolov8n-seg.pt"  # Path to the trained YOLOv8 model

model = YOLO(model_path)

from pathlib import Path

# Define the folder path
folder_path = "/home/seame/seame_img/"
# folder_path = "/home/seame/jetracer_stop_sign"

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    model.predict(file_path, save=True, imgsz=320, conf=0.5)

