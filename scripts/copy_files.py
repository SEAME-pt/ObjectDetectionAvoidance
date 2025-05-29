import os
import shutil

def copy_directory_files(source_dir, dest_dir, include_subdirs=True):
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Counter for copied files
    copied_count = 0
    
    if include_subdirs:
        # Copy entire directory tree
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            dst_path = os.path.join(dest_dir, item)
            
            try:
                if os.path.isdir(src_path):
                    # Copy directory and its contents
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"Copied directory: {item}")
                else:
                    # Copy individual file
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied file: {item}")
                    copied_count += 1
            except (shutil.Error, OSError) as e:
                print(f"Error copying {item}: {e}")
    else:
        # Copy only files in the source directory (not subdirectories)
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            if os.path.isfile(src_path):
                dst_path = os.path.join(dest_dir, item)
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied file: {item}")
                    copied_count += 1
                except (shutil.Error, OSError) as e:
                    print(f"Error copying {item}: {e}")
    
    print(f"Total files copied: {copied_count}")


def copy_matching_files(dir1, dir2, dest_dir):
    if not os.path.exists(dir1):
        print(f"Directory 1 does not exist: {dir1}")
        return
    if not os.path.exists(dir2):
        print(f"Directory 2 does not exist: {dir2}")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get list of files in both directories
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    
    # Find matching files (same name and extension)
    matching_files = files_dir1.intersection(files_dir2)
    
    if not matching_files:
        print("No matching files found between the two directories.")
        return
    
    # Counter for copied files
    copied_count = 0
    
    # Copy matching files from both directories
    for file in matching_files:
        src_path1 = os.path.join(dir1, file)
        src_path2 = os.path.join(dir2, file)
        dst_path = os.path.join(dest_dir, file)
        
        # Copy from dir1
        # if os.path.isfile(src_path1):
        #     try:
        #         shutil.copy2(src_path1, dst_path)
        #         print(f"Copied from dir1: {file}")
        #         copied_count += 1
        #     except (shutil.Error, OSError) as e:
        #         print(f"Error copying {file} from dir1: {e}")
        
        # Copy from dir2 (if not already copied due to same name)
        if os.path.isfile(src_path2) and not os.path.exists(dst_path):
            try:
                shutil.copy2(src_path2, dst_path)
                print(f"Copied from dir2: {file}")
                copied_count += 1
            except (shutil.Error, OSError) as e:
                print(f"Error copying {file} from dir2: {e}")
    
    print(f"Total files copied: {copied_count}")

# Example usage
if __name__ == "__main__":
    dir = "../cross/train/images"  
    destination_dir = "../dataset/images/train"  
    copy_directory_files(dir, destination_dir, include_subdirs=True)

    dir = "../cross/train/labels_seg"  
    destination_dir = "../dataset/labels/train"  
    copy_directory_files(dir, destination_dir, include_subdirs=True)

    # dir = "../dataset/datasets/speed/val/images" 
    # destination_dir = "../dataset/images/val"
    # copy_directory_files(dir, destination_dir, include_subdirs=True)

    # dir = "../dataset/datasets/speed/val/labels_seg" 
    # destination_dir = "../dataset/labels/val"
    # copy_directory_files(dir, destination_dir, include_subdirs=True)