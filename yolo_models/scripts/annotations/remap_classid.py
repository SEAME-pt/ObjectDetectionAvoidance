import os

def change_annotation_labels(label_dir, label_map):

    
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
            if parts and parts[0] in label_map:  # Check if line is not empty
                # Check if the first element (label) needs to be changed
                parts[0] = label_map[parts[0]]
                modified = True
                new_lines.append(' '.join(parts) + '\n')
            else:
                new_lines.append(line)
        
        # Write back to file only if modified
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1
 
    print(f"Total files modified: {modified_count}")

if __name__ == "__main__":
    label_map = {
        '2': '5',
        # Add other mappings if needed
    }
    output_directory = "../../8080/labels/"  # Adjust this path as needed
    change_annotation_labels(output_directory, label_map)

