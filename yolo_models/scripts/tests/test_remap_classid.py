# tests/test_label_utils.py
import os
from scripts.annotations.remap_classid import change_annotation_labels

def test_change_annotation_labels(tmp_path):
    # Setup test directory and files
    label_dir = tmp_path / "labels"
    label_dir.mkdir()

    # Create a test label file
    file_path = label_dir / "test1.txt"
    file_path.write_text("2 0.5 0.5 0.1 0.1\n3 0.6 0.6 0.1 0.1\n")
    label_map = {
        '2': '5',
    }
    # Run the function
    change_annotation_labels(str(label_dir), label_map)

    # Read the modified content
    result = file_path.read_text()
    
    # Assert the label was changed according to the mapping
    assert result == "5 0.5 0.5 0.1 0.1\n3 0.6 0.6 0.1 0.1\n"
