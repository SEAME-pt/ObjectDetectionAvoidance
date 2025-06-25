# YOLOv8 to TensorRT Conversion

This README provides a clear and concise guide to convert a YOLOv8 segmentation model (`.pt`) into a TensorRT engine (`.engine`).

---

## 📋 Prerequisites

- **Operating System**: Ubuntu (or similar)
- **Python 3** and **pip**
- **CUDA** and **TensorRT ≥ 8.0**
- **OpenCV ≥ 3.4.0**
- **PyTorch** and **ultralytics ≤ 8.2.103**
- Compatible **CUDA Toolkit** and **cuDNN**

---

## 🔧 Step-by-Step Instructions

### 1. Clone Repositories

```bash
# Ultralytics PyTorch implementation
git clone https://github.com/ultralytics/ultralytics.git

# YOLOv8 TensorRT code
https://github.com/wang-xinyu/tensorrtx/tree/master/yolov8
```

### 2. Generate the `.wts` File

```bash
# Copy the WTS generator script
git clone https://github.com/xiaocao-tian/yolov8_tensorrt.git
git clone https://github.com/ultralytics/ultralytics.git
cp yolov8_tensorrt/yolov8/gen_wts.py ultralytics/ultralytics/

# Run the conversion for segmentation (-t seg)
cd ultralytics/ultralytics
python3 gen_wts.py \
  -w /path/to/1706_best.pt \
  -o /path/to/1706_best.wts \
  -t seg
```

### 3. Set the Number of Classes

Open `tensorrtx/yolov8/include/config.h` and update:

```cpp
static const int kSegNumClass = <NUM_CLASSES>;
```

Replace `<NUM_CLASSES>` with the actual number of classes in your model.

### 4. Build TensorRT Project

```bash
# Go to the TensorRT YOLOv8 folder
cd tensorrtx/yolov8

# Clean and create a build directory
rm -rf build && mkdir build && cd build

# Copy the generated WTS file
cp ../../ultralytics/ultralytics/1706_best.wts .

# Generate build files and compile
cmake ..
make -j$(nproc)
```

### 5. Serialize to TensorRT Engine

```bash
# Convert .wts to .engine
docker
title: Serialize model to TensorRT plan file
sudo ./yolov8_seg \
  -s 1706_best.wts \
     1706_best.engine \
     n
```

***Note:**** Replace **`n`** with your model variant (**`n`**, **`s`**, **`m`**, **`l`**, **`x`**, etc.)*

### 6. Run Inference

1. Prepare a folder with test images:

   ```bash
   mkdir -p images
   cp /path/to/images/*.jpg images/
   ```

2. Run segmentation (CPU post-processing):

   ```bash
   ./yolov8_seg \
     -d 1706_best.engine \
     images \
     c \
     my_classes.txt
   ```

3. Or run segmentation (GPU post-processing):

   ```bash
   ./yolov8_seg \
     -d 1706_best.engine \
     images \
     g \
     my_classes.txt
   ```

Results will be saved in the `images/` folder with the prefix `seg_`.

---


