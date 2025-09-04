import os
import shutil
from scripts.annotations.bbox_seg import convert_bbox_to_segmentation

def test_convert_bbox_to_segmentation(tmp_path):
    # Setup fake input and output directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create a sample label file in YOLO bbox format
    label_file = input_dir / "sample.txt"
    with open(label_file, "w") as f:
        f.write("0 0.5 0.5 0.4 0.2\n")  # center_x, center_y, width, height

    # Run the conversion
    convert_bbox_to_segmentation(str(input_dir), str(output_dir))

    # Check if output file exists
    output_file = output_dir / "sample.txt"
    assert output_file.exists(), "Output file not created."

    # Check content of the output file
    with open(output_file, "r") as f:
        lines = f.readlines()

    parts = lines[0].strip().split() #whitespace
    assert parts[0] == "0"
    assert len(parts[1:]) == 8  # 4 points × 2 coordinates

    # Check that all coords are in range [0.0, 1.0]
    coords = list(map(float, parts[1:]))
    assert all(0.0 <= c <= 1.0 for c in coords), "Coords not in [0.0, 1.0]"

