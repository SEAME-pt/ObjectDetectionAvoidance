# Migration to New YOLO Model - best_202507181755.pt

## **Summary of Changes**

This document describes the changes made to migrate from the previous model to the new model `models/best_202507181755.pt`.

## **New Model Information**

### **Technical Characteristics**
- **File**: `models/best_202507181755.pt`
- **Type**: YOLOv8s-seg (segmentation)
- **Parameters**: 11,793,192
- **GFLOPs**: 42.7
- **Number of classes**: 8

### **Available Classes**
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | drivable | Drivable area |
| 1 | **lane** | **Road lane** |
| 2 | passadeira | Pedestrian crossing |
| 3 | stop sign | Stop sign |
| 4 | speed 50 | Speed limit 50 |
| 5 | speed 80 | Speed limit 80 |
| 6 | jetracer | JetRacer car |
| 7 | gate | Gate |

## **Changes Made**

### **1. Camera Script (`scripts/camera_yolo_to_shm.py`)**
```python
# BEFORE:
MODEL_PATH = "models/best.pt"
LANE_CLASS_ID = 80

# AFTER:
MODEL_PATH = "models/best_202507181755.pt"
LANE_CLASS_ID = 1  # 'lane' class in new model
```

### **2. Information Extraction Script (`scripts/extract_info_yolo_model.py`)**
```python
# BEFORE:
model_path = '/home/jetson/models_/best_202507181755.pt'

# AFTER:
model_path = '/home/jetson/Documents/e-codes/pid_final/pid_control/models/best_202507181755.pt'
```

## **Verifications Performed**

### **Loading Test**
- Model loads without errors
- Classes are recognized correctly
- 'lane' class is available with ID 1
- Predictions work normally

### **Compatibility**
- Same input structure (images)
- Same output structure (masks)
- Compatible with shared memory system
- Compatible with GStreamer pipeline

## **How to Use the New Model**

### **1. Run Complete System**
```bash
# Terminal 1: Start camera script
python3 scripts/camera_yolo_to_shm.py

# Terminal 2: Start PID control
make clean && make
./bin/jetracer_pid_controler
```

### **2. Test Model Only**
```bash
# Check model information
python3 scripts/extract_info_yolo_model.py
```

## **Recommended Settings**

### **For Camera Script**
```python
MODEL_PATH = "models/best_202507181755.pt"
LANE_CLASS_ID = 1  # 'lane' class
CONF_THRESHOLD = 0.25  # Can be adjusted as needed
```

### **Model Training Parameters**
- **Epochs**: 150
- **Batch size**: 16
- **Image size**: 320x320
- **Optimizer**: auto
- **Learning rate**: 0.002

## **Main Differences**

### **Previous vs. New Model**
| Aspect | Previous Model | New Model |
|--------|----------------|-----------|
| File name | `best.pt` | `best_202507181755.pt` |
| Lane class ID | 80 | 1 |
| Number of classes | Unknown | 8 |
| Training date | Unknown | 18/07/2025 |

## **Important Considerations**

### **1. Previous Model Backup**
- The previous model (`best.pt`) was not removed
- Can be restored by changing `MODEL_PATH` back

### **2. Performance**
- The new model may have different performance
- Adjust `CONF_THRESHOLD` if necessary
- Monitor detection quality

### **3. Additional Classes**
- The new model detects more classes (8 vs. previous)
- This may improve system robustness
- Classes like 'stop sign' and 'speed' may be useful in the future

## **Recommended Tests**

### **1. Basic Test**
- Verify system starts without errors
- Confirm masks are generated
- Validate PID control works

### **2. Performance Test**
- Compare FPS with previous model
- Check detection quality
- Test in different lighting conditions

### **3. Robustness Test**
- Test with different lane types
- Check behavior in curves
- Validate detection in adverse conditions

## **Next Steps**

1. **Run tests in real environment**
2. **Adjust parameters if necessary**
3. **Document observed performance**
4. **Consider future optimizations**

---

**Migration date**: 18/07/2025
**Status**: Completed and tested


