import os
import tempfile
from collections import Counter
from scripts.annotations.count_labels import count_class_ids  # Replace with the actual module name

def test_count_class_ids_basic():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample label files
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")

        with open(file1, "w") as f:
            f.write("1 0.5 0.5 0.1 0.1\n2 0.6 0.6 0.2 0.2\n")

        with open(file2, "w") as f:
            f.write("1 0.1 0.1 0.3 0.3\n3 0.9 0.9 0.2 0.2\n")

        result = count_class_ids(temp_dir)

        expected = {
            '1': 2,
            '2': 1,
            '3': 1
        }

        assert result == expected

def test_count_class_ids_empty_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file1 = os.path.join(temp_dir, "empty.txt")
        open(file1, "w").close()  # Create empty file
        result = count_class_ids(temp_dir)
        assert result == {}
