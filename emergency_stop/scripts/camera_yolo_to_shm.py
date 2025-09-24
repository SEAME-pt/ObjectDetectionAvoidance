# File: camera_yolo_to_shm.py
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from multiprocessing import shared_memory

# ===== Configurations =====
IMG_WIDTH = 128
IMG_HEIGHT = 128
SHM_NAME = "mask_shared"
MODEL_PATH = "models/best_202507181755.pt"
LANE_CLASS_ID = 1
DRIVABLE_CLASS_ID = 0
CONF_THRESHOLD = 0.25
SAVE_DIR = "masks"

# ===== CSI camera GStreamer pipeline =====
PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=320, height=240, format=NV12, framerate=15/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)

# ===== Initialize shared memory (flag + 2 images) =====
TOTAL_SIZE = 1 + 2 * IMG_WIDTH * IMG_HEIGHT  # 1 byte for flag + 2 masks
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=TOTAL_SIZE)
except FileExistsError:
    existing = shared_memory.SharedMemory(name=SHM_NAME)
    existing.close()
    existing.unlink()
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=TOTAL_SIZE)

flag_buf = np.ndarray((1,), dtype=np.uint8, buffer=shm.buf, offset=0)
lane_mask_buf = np.ndarray((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8, buffer=shm.buf, offset=1)
drivable_mask_buf = np.ndarray((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8, buffer=shm.buf, offset=1 + IMG_WIDTH * IMG_HEIGHT)

# ===== Load YOLO model and camera =====
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

#for _ in range(5):
#    cap.read()
#    time.sleep(0.05)

if not cap.isOpened():
    print("Error opening CSI camera.")
    shm.close()
    shm.unlink()
    exit(1)

os.makedirs(SAVE_DIR, exist_ok=True)
print("Camera and model loaded. Press ESC to exit.")

try:
    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Frame not captured.")
            break

        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        lane_mask_final = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        drivable_mask_final = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if results[0].masks is not None and results[0].boxes is not None:
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, cls_id in enumerate(classes):
                try:
                    mask_i = cv2.resize(masks[i], (frame.shape[1], frame.shape[0]))
                    mask_i = (mask_i > 0.5).astype(np.uint8)

                    if cls_id == LANE_CLASS_ID:
                        lane_mask_final = np.logical_or(lane_mask_final, mask_i)
                    elif cls_id == DRIVABLE_CLASS_ID:
                        drivable_mask_final = np.logical_or(drivable_mask_final, mask_i)

                except Exception as e:
                    print(f"Error processing mask {i}: {e}")

            lane_mask_final = (lane_mask_final * 255).astype(np.uint8)
            drivable_mask_final = (drivable_mask_final * 255).astype(np.uint8)

        lane_mask_resized = cv2.resize(lane_mask_final, (IMG_WIDTH, IMG_HEIGHT))
        drivable_mask_resized = cv2.resize(drivable_mask_final, (IMG_WIDTH, IMG_HEIGHT))

        # === Synchronization: wait for C++ to process (flag == 0) ===
        while flag_buf[0] != 0:
            time.sleep(0.001)

        # === Send masks and signal (flag = 1) ===
        lane_mask_buf[:] = lane_mask_resized[:]
        drivable_mask_buf[:] = drivable_mask_resized[:]
        flag_buf[0] = 1

        fps = 1 / (time.time() - start)
        print(f"FPS: {fps:.2f}")

        #cv2.imshow("CSI Camera", frame)
        #cv2.imshow("Lane Mask", lane_mask_resized)
        #cv2.imshow("Drivable Mask", drivable_mask_resized)

        if cv2.waitKey(1) == 27:
            break

finally:
    cap.release()
    shm.close()
    shm.unlink()
    cv2.destroyAllWindows()
    print("Closed.")
