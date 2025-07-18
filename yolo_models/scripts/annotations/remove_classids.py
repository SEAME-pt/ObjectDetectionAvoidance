import os
import sys



def remove_classes_from_file(file_path, class_ids_to_remove):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Keep lines that don't start with any of the class IDs
    new_lines = []
    for line in lines:
        if not line.strip():
            continue  # Skip empty lines
        class_id = line.split()[0]
        # print(class_id)
        if class_id not in class_ids_to_remove:
            print(class_id)
            new_lines.append(line)

    # Overwrite file
    with open(file_path, "w") as file:
        file.writelines(new_lines)

def process_directory(dir_path, class_ids_to_remove):
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if fname.endswith(".txt"):
                file_path = os.path.join(root, fname)
                remove_classes_from_file(file_path, class_ids_to_remove)
                # print(f"Processed: {file_path}")
                
                

def process_directory2(directory, keep_if_present, remove_if_present):
    keep_if_present = set(str(cid) for cid in keep_if_present)
    remove_if_present = set(str(cid) for cid in remove_if_present)

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r") as file:
            lines = file.readlines()

        # Check if any class ID to keep is present
        class_ids_in_file = {line.split()[0] for line in lines if line.strip()}

        # class_ids_in_file = {line.split()[0] for line in lines}
        if not class_ids_in_file & keep_if_present:
            continue  # skip file if none of the keep IDs are present

        # Remove lines with class IDs we want to remove
        new_lines = [line for line in lines if line.strip() and line.split()[0] not in remove_if_present]

        # new_lines = [line for line in lines if line.split()[0] not in remove_if_present]

        # Overwrite file
        with open(filepath, "w") as file:
            file.writelines(new_lines)




if __name__ == "__main__":
    # process_directory2("../split_dataset/val/labels", ["3", "4", "12", "13", "17"], ["2", "3"])
    process_directory("../split_dataset/val/labels", ["0", "1", "5", "6", "9", "13", "17"])