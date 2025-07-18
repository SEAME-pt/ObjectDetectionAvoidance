import os
import cv2
import random
import re
import shutil
import sys


def remove_unmatched_txt(label_dir, image_dir):
    txt_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    jpg_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

    removed_count = 0

    for txt_file in txt_files:
        if txt_file not in jpg_files:
            txt_path = os.path.join(label_dir, txt_file + '.txt')
            print(f"Checking files: {txt_file}")
            if os.path.exists(txt_path):
                os.remove(txt_path)
                removed_count += 1

    print(f"Total txt removed: {removed_count}")
    removed_count = 0
    for jpg_file in jpg_files:
        if jpg_file not in txt_files:
            print(f"Checking files: {jpg_file}")
            jpg_path = os.path.join(image_dir, jpg_file + '.jpg')
            if os.path.exists(jpg_path):
                # os.remove(jpg_path)
                removed_count += 1
    
    print(f"Total img removed: {removed_count}")



if __name__ == "__main__":
    # label_directory = "../dataset/labels/train"  # Replace with your label directory path
    # image_directory = "../dataset/images/train"  # Replace with your image directory path
    # label_directory = "../clutter/newww/labels/train"  # Replace with your label directory path
    # image_directory = "../clutter/newww/images/train"
    # remove_unmatched_txt(label_directory, image_directory)
