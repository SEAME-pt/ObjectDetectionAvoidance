import os
import shutil

def copy_files_with_class_ids(txt_dir, img_dir, output_dir, target_class_ids=[2, 4]):
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Get all txt files
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        
        # Read txt file and check for target class IDs
        has_target_class = False
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) in target_class_ids and int(parts[0]) != 12:
                    has_target_class = True
                    break
        
        if has_target_class:
            # Copy txt file
            shutil.copy(txt_path, os.path.join(output_dir, 'labels', txt_file))
            print(f"Copied txt: {txt_file}")
            
            # Copy corresponding image (try common extensions)
            base_name = os.path.splitext(txt_file)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                img_file = base_name + ext
                img_path = os.path.join(img_dir, img_file)
                if os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(output_dir, 'images', img_file))
                    print(f"Copied image: {img_file}")
                    break
            else:
                print(f"No image found for {txt_file}")

# Example usage
txt_dir = '../dataset/labels/train'  # Directory with txt files
img_dir = '../dataset/images/train/'   # Directory with images
shutil.rmtree('../filtered/', ignore_errors=True)  # Clear previous output
output_dir = '../filtered/'  # Output directory
copy_files_with_class_ids(txt_dir, img_dir, output_dir)