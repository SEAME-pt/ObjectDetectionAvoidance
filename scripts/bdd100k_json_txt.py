import os
import json
from PIL import Image
import os
import json
import shutil

shutil.rmtree("./txt_train", ignore_errors=True)  # Clear previous output
def load_annotations(json_dir, output_dir="./val"):

    os.makedirs(output_dir, exist_ok=True)

    class_id_map = {
        6508800: 0,  # red
        6508801: 1,  # yellow
        6508802: 2,  # green
        6508803: 3,  # blue
        6508804: 4,  # purple
        6508805: 5,  # orange
        6508806: 6,  # brown
        6508807: 7,  # pink
        6508808: 8,  # gray
        6508809: 9,  # black
        6508810: 10, # white
        6508811: 11, # cyan
        6508812: 12, # magenta
        # Adjust based on your dataset
    }

    for file_name in os.listdir(json_dir):
        if not file_name.endswith(".json"):
            print(f"Skipping non-JSON file: {file_name}")
            continue

        json_path = os.path.join(json_dir, file_name)

        with open(json_path, "r") as f:
            data = json.load(f)

        img_width = data["size"]["width"]
        img_height = data["size"]["height"]
        objects = data.get("objects", [])

        yolo_lines = []

        for obj in objects:
            class_id_raw = obj["classId"]
            if class_id_raw in class_id_map:
                class_id = class_id_map[class_id_raw]
            else:
                class_id = obj["classId"]

            exterior = obj["points"]["exterior"]

            points = obj["points"]["exterior"]
            if not points:
                print(f"Skipping empty points for class {class_id}")
                continue

            # Convert rectangle to polygon (4 points)
            if obj.get("geometryType") == "rectangle":
                if len(points) != 2:
                    print(f"Invalid rectangle for class {class_id}, expected 2 points, got {len(points)}")
                    continue
                (x1, y1), (x2, y2) = points
                points = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]

            if len(points) == 2:
                (x1, y1), (x2, y2) = points
                xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                points = [
                    [x1, y1],
                    [x2, y2],
                    [xm, ym],
                ]
            if len(points) < 3:
                print(f"Skipping object for class {class_id} with insufficient points: {len(points)}")
                continue  # YOLOv8 requires at least a triangle

            # Normalize points
            coords = []
            for x, y in points:
                coords.append(f"{x / img_width:.6f}")
                coords.append(f"{y / img_height:.6f}")

            yolo_line = f"{class_id} " + " ".join(coords)
            yolo_lines.append(yolo_line)

        # Write output .txt
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, base_name + ".txt")

        with open(output_path, "w") as out_f:
            out_f.write("\n".join(yolo_lines))



images_dir = "/home/seame/Downloads/bdd100k/val/img"              # Where the images are

json_path = '/home/seame/Downloads/bdd100k/val/ann'

load_annotations(json_path)

