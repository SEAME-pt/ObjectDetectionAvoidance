import cv2
import numpy as np
import os
from ultralytics import YOLO
import shutil
from pathlib import Path

model_path = "../models/seame_training_50/weights/best.pt"  

model = YOLO(model_path)
results = model.predict("../clutter/video_seame.avi", save=True, conf=0.1)

# image_dir = "../clutter/signal_lab_test"

# for file_name in os.listdir(image_dir):
#     file_path = os.path.join(image_dir, file_name)
#     if os.path.isfile(file_path):
#         model.predict(file_path, save=True, imgsz=320, conf=0.1)

# os.system('yolo val model="../models/seame_training/weights/best.pt" data="/home/seame/ObjectDetectionAvoidance/split_dataset/data.yaml"')
# # os.system('yolo predict model="../models/yolo-object-lane2/weights/best.pt" source="/home/seame/frames/frames2/frame_0058.jpg \
# #     /home/seame/frames/frames2/frame_0092.jpg /home/seame/frames/frames2/frame_0026.jpg /home/seame/frames/frames2/frame_0035.jpg \
# #     /home/seame/frames/frames2/frame_0033.jpg" project=predictions name=lane2')

# os.system('yolo val model="../models/second_training2/weights/best.pt" data="/home/seame/ObjectDetectionAvoidance/second_training/data.yaml"')
# os.system('yolo predict model="../models/yolo-object-lane2/weights/best.pt" source="/home/seame/frames/frames2/frame_0058.jpg \
#     /home/seame/frames/frames2/frame_0092.jpg /home/seame/frames/frames2/frame_0026.jpg /home/seame/frames/frames2/frame_0035.jpg \
#     /home/seame/frames/frames2/frame_0033.jpg" project=predictions name=lane')

# os.system('yolo val model="../models/yolo-object-lane-unfroze/weights/best.pt" data="/home/seame/ObjectDetectionAvoidance/dataset/data.yaml"')
# os.system('yolo predict model="../models/yolo-object-lane2/weights/best.pt" source="/home/seame/frames/frames2/frame_0058.jpg \
#     /home/seame/frames/frames2/frame_0092.jpg /home/seame/frames/frames2/frame_0026.jpg /home/seame/frames/frames2/frame_0035.jpg \
#     /home/seame/frames/frames2/frame_0033.jpg" project=predictions name=froze')
