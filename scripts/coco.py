import os
import random
import shutil
from ultralytics.utils.downloads import download
from pathlib import Path

# yaml = {"path": "/home/seame/ObjectDetectionAvoidance/dataset/coco"}
# dir = Path(yaml["path"])  

# segments = True  
# url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
# urls = [url + "coco2017labels-segments.zip"] 
# download(urls, dir=dir.parent)

# urls = [
#     "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
#     "http://images.cocodataset.org/zips/val2017.zip",    # 1G, 5k images
# ]
# download(urls, dir=dir / "images", threads=3)

coco_images = "../dataset/coco/images/train"
coco_labels = "../dataset/coco/labels/train"
subset_dir = "../dataset/coco_subset/"
os.makedirs(subset_dir, exist_ok=True)
os.makedirs(os.path.join(subset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(subset_dir, "labels"), exist_ok=True)

all_images = os.listdir(coco_images)
subset_size = 10000
subset_images = random.sample(all_images, subset_size)

for img in subset_images:
    shutil.copy(os.path.join(coco_images, img), os.path.join(subset_dir, "images", img))
    label = img.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(coco_labels, label)):
        shutil.copy(os.path.join(coco_labels, label), os.path.join(subset_dir, "labels", label))