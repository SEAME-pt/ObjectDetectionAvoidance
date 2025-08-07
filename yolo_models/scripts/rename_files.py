import os
import re

def rename_images_labels(dir_path_images, dir_path_labels):
    image_files = [f for f in os.listdir(dir_path_images) if os.path.isfile(os.path.join(dir_path_images, f))]
    existing_images = set(image_files)

    for filename in image_files:
        name, ext = os.path.splitext(filename)
        full_image_path = os.path.join(dir_path_images, filename)

        # Skip non-image files
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # Clean up base name
        new_base = re.sub(r'(_jpg.*)+$', '', name)
        candidate = new_base + ext
        counter = 1
        existing_images.remove(filename)

        # Avoid name collisions
        while candidate in existing_images:
            candidate = f"{new_base}_{counter}{ext}"
            counter += 1

        existing_images.add(candidate)

        if filename != candidate:
            # Rename image
            new_image_path = os.path.join(dir_path_images, candidate)
            os.rename(full_image_path, new_image_path)
            print(f"Renamed image: {filename} -> {candidate}")

            # Rename corresponding .txt file
            old_txt = os.path.join(dir_path_labels, name + '.txt')
            new_txt = os.path.join(dir_path_labels, os.path.splitext(candidate)[0] + '.txt')
            print(f"Renaming label: {name}.txt to {os.path.splitext(candidate)[0]}.txt")
            if os.path.exists(old_txt):
                os.rename(old_txt, new_txt)
                print(f"Renamed label: {name}.txt -> {os.path.splitext(candidate)[0]}.txt")


# image = "../8080/train/"
# label = "../8080/labels"
# rename_images_labels(image, label)
