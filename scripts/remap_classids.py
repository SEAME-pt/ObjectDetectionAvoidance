import os

def change_annotation_labels(label_dir):
    # Mapping of old labels to new labels
    label_map = {
                '1': '0',  # Assuming '1' is the background class
                '2': '1',
                '3': '10',
                '4': '11',
                '5': '12', 
                '6': '13',
                '7': '14',
                '8': '2',
                '9': '3',
                '10': '4',
                '11': '5',
                '12': '8',
                '13': '9',

                }
    
    # Get list of txt files in label directory
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # Counter for modified files
    modified_count = 0
    
    for txt_file in txt_files:
        file_path = os.path.join(label_dir, txt_file)
        modified = False
        new_lines = []
        
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Process each line
        for line in lines:
            parts = line.strip().split()
            if parts:  # Check if line is not empty
                # Check if the first element (label) needs to be changed
                if parts[0] in label_map:
                    parts[0] = label_map[parts[0]]
                    modified = True
                    new_lines.append(' '.join(parts) + '\n')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Write back to file only if modified
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1
            # print(f"Modified: {txt_file}")

        # else:
        #     base_name = os.path.splitext(txt_file)[0]
        #     image_dir = '../dataset/images/train'  # Adjust this path as needed
        #     image_path = os.path.join(image_dir, base_name + '.jpg')
        #     os.remove(file_path)
        #     if os.path.exists(image_path):
        #         # os.remove(image_path)
        #         print(f"Removed: {txt_file} and corresponding image {base_name}.jpg")
        #     else:
        #         print(f"No corresponding image found for {txt_file}")
    print(f"Total files modified: {modified_count}")

if __name__ == "__main__":
    output_directory = "../clutter/new/output" 
    change_annotation_labels(output_directory)

    # label_directory = "../speed/val/labels_seg"  
    # change_annotation_labels(label_directory)
