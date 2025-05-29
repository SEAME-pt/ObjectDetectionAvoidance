import os

def convert_bbox_to_segmentation(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of txt files in label directory
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    # Counter for processed files
    processed_count = 0
    
    for txt_file in txt_files:
        input_path = os.path.join(label_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file)
        
        # Read the input annotation file
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:  # Skip invalid lines
                print(f"Skipping invalid line in {txt_file}: {line.strip()}")
                new_lines.append(line)
                continue
            
            try:
                # Parse bounding box annotation
                class_id = parts[0]
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Calculate the four corners of the bounding box
                half_width = width / 2
                half_height = height / 2
                x1 = center_x - half_width  # Top-left x
                y1 = center_y - half_height  # Top-left y
                x2 = center_x + half_width  # Top-right x
                y2 = center_y - half_height  # Top-right y
                x3 = center_x + half_width  # Bottom-right x
                y3 = center_y + half_height  # Bottom-right y
                x4 = center_x - half_width  # Bottom-left x
                y4 = center_y + half_height  # Bottom-left y
                
                # Ensure coordinates are within [0, 1]
                coords = [x1, y1, x2, y2, x3, y3, x4, y4]
                coords = [max(0.0, min(1.0, coord)) for coord in coords]
                
                # Format the new segmentation line
                new_line = f"{class_id} {' '.join(map(str, coords))}\n"
                new_lines.append(new_line)
                
            except ValueError:
                print(f"Error parsing line in {txt_file}: {line.strip()}")
                new_lines.append(line)
                continue
        
        # Write to output file
        with open(output_path, 'w') as f:
            f.writelines(new_lines)
        processed_count += 1
        print(f"Processed: {txt_file}")
    
    print(f"Total files processed: {processed_count}")


if __name__ == "__main__":
    label_directory = "../speed/train/labels"  # Path to the label directory
    output_directory = "../speed/train/labels_seg"  
    convert_bbox_to_segmentation(label_directory, output_directory)
