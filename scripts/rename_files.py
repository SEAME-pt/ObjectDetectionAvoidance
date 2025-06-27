# import os
# from collections import defaultdict
# import shutil

# def rename_files_by_prefix(directory, prefix_length=13, max_per_group=3):
#     # Validate directory
#     if not os.path.isdir(directory):
#         print(f"Error: {directory} is not a valid directory")
#         return

#     # Collect files and group by first 13 characters
#     file_groups = defaultdict(list)
#     for filename in os.listdir(directory):
#         if os.path.isfile(os.path.join(directory, filename)):
#             # Extract prefix (first 13 characters, or less if filename is shorter)
#             prefix = filename[:min(prefix_length, len(filename))]
#             file_groups[prefix].append(filename)

#     print(f"Found {len(file_groups)} groups of files based on first {prefix_length} characters")

#     # Process each group
#     renamed_count = 0
#     for prefix, filenames in file_groups.items():
#         print(f"\nProcessing group with prefix '{prefix}' ({len(filenames)} files): {filenames}")

#         if len(filenames) == 0:
#             continue

#         # Sort files to ensure consistent renaming order
#         filenames.sort()

#         # Limit to max_per_group files
#         files_to_rename = filenames[:max_per_group]
#         if len(filenames) > max_per_group:
#             print(f"Warning: Group '{prefix}' has {len(filenames)} files, but only renaming {max_per_group}")

#         for index, old_name in enumerate(files_to_rename, start=1):
#             # Split filename into name and extension
#             base_name, ext = os.path.splitext(old_name)
#             # Create new filename: prefix + counter + extension
#             new_name = f"{prefix}_{index}{ext}"
#             old_path = os.path.join(directory, old_name)
#             new_path = os.path.join(directory, new_name)

#             # Check for conflicts
#             if os.path.exists(new_path) and new_path != old_path:
#                 print(f"Error: Cannot rename '{old_name}' to '{new_name}' (file already exists)")
#                 continue

#             try:
#                 # Rename file
#                 shutil.move(old_path, new_path)
#                 print(f"Renamed: '{old_name}' -> '{new_name}'")
#                 renamed_count += 1
#             except Exception as e:
#                 print(f"Error renaming '{old_name}' to '{new_name}': {e}")

#     print(f"\nTotal files renamed: {renamed_count}")

# # Example usage
# if __name__ == "__main__":
#     # Replace with your directory paths
#     image_dir = "../dataset/seame/annotations/test"
#     # label_dir = "../dataset/seame_coco/annotations/train"

#     print("Renaming image files...")
#     rename_files_by_prefix(image_dir, prefix_length=13, max_per_group=3)

import os
import shutil
import re
from collections import defaultdict

def clean_filename(filename):
    """
    Removes '.rf' and 3 characters before it, and anything after.
    """
    base, _ = os.path.splitext(filename)
    idx = base.find(".rf")
    if idx != -1 and idx >= 3:
        return base[:idx - 3]
    return base

def rename_files_and_labels(images_dir, labels_dir):
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print("Error: Provided directories do not exist.")
        return

    new_name_counts = defaultdict(int)
    renamed_files = 0

    for fname in os.listdir(images_dir):
        old_img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(old_img_path):
            continue

        # Step 1: Clean name
        name_without_rf = clean_filename(fname)

        # Step 2: Handle duplicates by appending counter
        count = new_name_counts[name_without_rf]
        while True:
            suffix = f"_{count}" if count > 0 else ""
            new_base_name = f"{name_without_rf}{suffix}"
            new_img_name = f"{new_base_name}.jpg"
            new_img_path = os.path.join(images_dir, new_img_name)
            new_lbl_name = f"{new_base_name}.txt"
            new_lbl_path = os.path.join(labels_dir, new_lbl_name)

            if not os.path.exists(new_img_path):
                break
            count += 1

        new_name_counts[name_without_rf] = count + 1

        # Step 3: Rename image
        try:
            shutil.move(old_img_path, new_img_path)
            # print(f"Renamed image: {fname} -> {new_img_name}")
        except Exception as e:
            print(f"Failed to rename {fname}: {e}")
            continue

        # Step 4: Rename corresponding label
        old_txt_name = os.path.splitext(fname)[0] + ".txt"
        old_lbl_path = os.path.join(labels_dir, old_txt_name)
        if os.path.exists(old_lbl_path):
            try:
                shutil.move(old_lbl_path, new_lbl_path)
                # print(f"Renamed label: {old_txt_name} -> {new_lbl_name}")
            except Exception as e:
                print(f"Failed to rename label {old_txt_name}: {e}")
        else:
            print(f"Warning: Label file not found for {fname} ({old_txt_name})")

        renamed_files += 1

    print(f"\nTotal renamed files: {renamed_files}")

def rename_files_to_single_suffix_1(dir_path):
    files = os.listdir(dir_path)
    existing_names = set(os.listdir(dir_path))  # Track existing filenames

    for filename in files:
        full_path = os.path.join(dir_path, filename)
        if not os.path.isfile(full_path):
            continue

        name, ext = os.path.splitext(filename)

        # Remove any trailing _1 or multiple _1's (like _1__1_1 etc)
        # We'll remove any trailing sequence of (_1)+, possibly separated by underscores
        # if (name.endswith('_4')):
        new_base = re.sub(r'(.jpg)+$', '', name)  # Remove trailing _1 repeated
        # new_base = re.sub(r'(_4.*)$', '', name)
        # Now add a single _1 at the end
        # new_base = new_base + "_1"
        # new_filename = new_base + ext

        # If the file name already exists, add _2, _3, etc.
        counter = 1
        # candidate = new_filename
        # while candidate in existing_names:
        #     print(f"File {candidate} already exists, trying with counter {counter}")
        #     counter += 1
            # candidate = f"{new_base[:-2]}_{counter}{ext}"  # remove old _1 and add _2, _3 etc

        # Rename file
        old_path = full_path
        new_path = os.path.join(dir_path, new_base + ext)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_path}")

        # Update existing files set
        # existing_names.add(candidate)
        existing_names.discard(filename)


directory_path = "../dataset/labels/val"
rename_files_to_single_suffix_1(directory_path)
# directory_path = "../clutter/corrects/output"
# rename_files_to_single_suffix_1(directory_path)


# # Example usage
# if __name__ == "__main__":
#     images_dir = "../corrects/train"
#     labels_dir = "../corrects/output"
#     rename_files_and_labels(images_dir, labels_dir)

    
