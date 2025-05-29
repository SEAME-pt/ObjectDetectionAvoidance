import os
import random
import re

def delete_random_short_basename_images(image_dir, num_to_delete=1000, max_basename_length=6):
    # Ensure directory exists
    if not os.path.exists(image_dir):
        print(f"Image directory does not exist: {image_dir}")
        return

    # Get list of jpg files with basename length <= max_basename_length
    jpg_files = [
        f for f in os.listdir(image_dir)
        if f.endswith('.jpg') and re.match(r'^[a-zA-Z0-9]{8}-[a-zA-Z0-9]{8}$', os.path.splitext(f)[0])
    ]
    
    # Check if there are enough files to delete
    total_eligible = len(jpg_files)
    print(f"Total .jpg files found: {total_eligible}")
    if total_eligible == 0:
        print("No .jpg files found")
        return

    num_to_delete = min(num_to_delete, total_eligible)
    print(f"Found {total_eligible} eligible .jpg files. Will delete {num_to_delete} of them.")

    # Randomly select files to delete
    files_to_delete = random.sample(jpg_files, num_to_delete)
    
    # Counter for deleted files
    deleted_count = 0
    
    # Delete selected files
    for file in files_to_delete:
        file_path = os.path.join(image_dir, file)
        try:
            # os.remove(file_path)
            print(f"Deleted: {file}")
            deleted_count += 1
        except OSError as e:
            print(f"Error deleting file {file}: {e}")
    
    print(f"Total files deleted: {deleted_count}")

def remove_unmatched_txt(label_dir, image_dir):
    txt_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    jpg_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

    removed_count = 0

    for txt_file in txt_files:
        if txt_file not in jpg_files:
            txt_path = os.path.join(label_dir, txt_file + '.txt')
            if os.path.exists(txt_path):
                os.remove(txt_path)
                removed_count += 1

    print(f"Total files removed: {removed_count}")
    removed_count = 0
    for jpg_file in jpg_files:
        if jpg_file not in txt_files:
            # print(f"Checking files: {jpg_file}")
            jpg_path = os.path.join(image_dir, jpg_file + '.jpg')
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
                removed_count += 1
    
    print(f"Total files removed: {removed_count}")

# Example usage
if __name__ == "__main__":
    label_directory = "../dataset/labels/train"  # Replace with your label directory path
    image_directory = "../dataset/images/train"  # Replace with your image directory path
    # delete_random_short_basename_images(image_directory)
    remove_unmatched_txt(label_directory, image_directory)