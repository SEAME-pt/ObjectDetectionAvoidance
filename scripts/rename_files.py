import os
from collections import defaultdict
import shutil

def rename_files_by_prefix(directory, prefix_length=13, max_per_group=3):
    # Validate directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return

    # Collect files and group by first 13 characters
    file_groups = defaultdict(list)
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # Extract prefix (first 13 characters, or less if filename is shorter)
            prefix = filename[:min(prefix_length, len(filename))]
            file_groups[prefix].append(filename)

    print(f"Found {len(file_groups)} groups of files based on first {prefix_length} characters")

    # Process each group
    renamed_count = 0
    for prefix, filenames in file_groups.items():
        print(f"\nProcessing group with prefix '{prefix}' ({len(filenames)} files): {filenames}")

        if len(filenames) == 0:
            continue

        # Sort files to ensure consistent renaming order
        filenames.sort()

        # Limit to max_per_group files
        files_to_rename = filenames[:max_per_group]
        if len(filenames) > max_per_group:
            print(f"Warning: Group '{prefix}' has {len(filenames)} files, but only renaming {max_per_group}")

        for index, old_name in enumerate(files_to_rename, start=1):
            # Split filename into name and extension
            base_name, ext = os.path.splitext(old_name)
            # Create new filename: prefix + counter + extension
            new_name = f"{prefix}_{index}{ext}"
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)

            # Check for conflicts
            if os.path.exists(new_path) and new_path != old_path:
                print(f"Error: Cannot rename '{old_name}' to '{new_name}' (file already exists)")
                continue

            try:
                # Rename file
                shutil.move(old_path, new_path)
                print(f"Renamed: '{old_name}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming '{old_name}' to '{new_name}': {e}")

    print(f"\nTotal files renamed: {renamed_count}")

# Example usage
if __name__ == "__main__":
    # Replace with your directory paths
    image_dir = "../dataset/seame/annotations/test"
    # label_dir = "../dataset/seame_coco/annotations/train"

    print("Renaming image files...")
    rename_files_by_prefix(image_dir, prefix_length=13, max_per_group=3)
