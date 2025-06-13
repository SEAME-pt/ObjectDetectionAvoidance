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
    # Validate input directories
    if not os.path.exists(dir1):
        print(f"Directory 1 does not exist: {dir1}")
        return
    if not os.path.exists(dir2):
        print(f"Directory 2 does not exist: {dir2}")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Created/Verified destination directory: {dest_dir}")
    
    # Get list of files in both directories with base names
    files_dir1 = {os.path.splitext(f)[0]: f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files_dir2 = {os.path.splitext(f)[0]: f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}
    
    # Debug: List files
    print(f"Files in {dir1}: {list(files_dir1.values())}")
    print(f"Files in {dir2}: {list(files_dir2.values())}")
    
    # Find matching base filenames
    matching_bases = set(files_dir1.keys()) & set(files_dir2.keys())
    print(f"Found {len(matching_bases)} matching base filenames: {matching_bases}")
    
    # Copy matching files to dest_dir
    copied_files = 0
    for base in matching_bases:
        file1 = files_dir1[base]
        file2 = files_dir2[base]
        
        src1 = os.path.join(dir1, file1)
        src2 = os.path.join(dir2, file2)
        dst1 = os.path.join(dest_dir, file1)
        dst2 = os.path.join(dest_dir, file2)
        
        try:
            # shutil.copy2(src1, dst1)  # Preserves metadata
            shutil.copy2(src2, dst2)
            print(f"Copied: {file1} -> {dst1}")
            print(f"Copied: {file2} -> {dst2}")
            copied_files += 1
        except Exception as e:
            print(f"Error copying {file1} or {file2}: {e}")
    
    print(f"Total files copied: {copied_files}")


if __name__ == "__main__":
    # dir = "../da_seame/train/"  
    # destination_dir = "../dataset/images/train"  
    # copy_directory_files(dir, destination_dir, include_subdirs=True)

    dir1 = "./chosen/"
    dir2 = "/home/seame/frames/frames3"  
    destination_dir = "./chosen/images"  
    copy_matching_files(dir1, dir2, destination_dir)
    # copy_directory_files(dir, destination_dir, include_subdirs=True)

