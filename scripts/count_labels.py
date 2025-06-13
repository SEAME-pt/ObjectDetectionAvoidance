import os
from collections import Counter

def count_class_ids(label_dir):
    if not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
        return {}
    
    # Initialize counter for class IDs
    class_counts = Counter()
    
    # Get list of txt files
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in {label_dir}")
        return {}
    
    # Process each annotation file
    for txt_file in txt_files:
        file_path = os.path.join(label_dir, txt_file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Count class IDs in each line
            for line in lines:
                parts = line.strip().split()
                if parts:  # Check if line is not empty
                    try:
                        class_id = parts[0]  # Class ID is the first element
                        class_counts[class_id] += 1
                    except (IndexError, ValueError):
                        print(f"Invalid line in {txt_file}: {line.strip()}")
                        continue
        except OSError as e:
            print(f"Error reading file {txt_file}: {e}")
            continue
    
    # Print results
    if class_counts:
        print("\nClass ID Counts:")
        for class_id, count in sorted(class_counts.items()):
            print(f"Class ID {class_id}: {count} occurrences")
        print(f"Total annotations: {sum(class_counts.values())}")
        print(f"Total files processed: {len(txt_files)}")
    else:
        print("No valid annotations found.")
    
    return dict(class_counts)

# Example usage
if __name__ == "__main__":
    label_directory = "../new/output"  # Replace with your label directory path
    class_counts = count_class_ids(label_directory)