import os
import json
import tempfile
from scripts.annotations.coco_txt import coco_to_txt  # adjust import

def make_coco_json(path):
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 320, "height": 320}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "segmentation": [[10,10, 50,10, 50,50, 10,50]],
                "bbox": [10,10,40,40]
            }
        ],
        "categories": [{"id":0, "name":"cat"}]
    }
    with open(path, 'w') as f:
        json.dump(coco, f)

def test_coco_to_txt_basic(tmp_path):
    json_path = tmp_path / "test.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    make_coco_json(json_path)

    coco_to_txt(str(json_path), str(output_dir))

    # Check output TXT file exists
    txt_file = output_dir / "img1.txt"
    assert txt_file.exists()

    # Check contents: has class id and coordinates
    with open(txt_file) as f:
        lines = f.readlines()
    assert len(lines) == 1
    parts = lines[0].strip().split()
    assert parts[0] == "0"  # class id
    coords = list(map(float, parts[1:]))
    assert all(0.0 <= c <= 1.0 for c in coords)

def test_coco_to_txt_no_annotations(tmp_path):
    json_path = tmp_path / "test.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # COCO JSON with no annotations
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 320, "height": 320}],
        "annotations": [],
        "categories": []
    }
    with open(json_path, 'w') as f:
        json.dump(coco, f)

    coco_to_txt(str(json_path), str(output_dir))

    # No txt file should be created because no annotations
    assert not any(output_dir.iterdir())

# Add tests for invalid polygons, malformed JSON, etc.
