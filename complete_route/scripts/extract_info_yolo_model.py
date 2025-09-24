from ultralytics import YOLO
import torch
import os

# Model path
model_path = '../models/best_202507181755.pt'

# Check if file exists
if not os.path.exists(model_path):
    print(f"[ERROR] File '{model_path}' not found.")
    exit(1)

print(f"\n[INFO] Loading model: {model_path}")
model = YOLO(model_path)

# Basic information
print("\nBasic model information:")
print(f"- Model type: {type(model)}")
print(f"- Number of classes: {model.model.nc}")
print(f"- Class names (model.names): {model.names}")

# Model structure
print("\nModel structure (summary):")
model.info(verbose=True)

# Training arguments and hyperparameters
print("\nTraining arguments and hyperparameters available in the model:")
try:
    args = model.model.args
    for k, v in vars(args).items():
        print(f"  - {k}: {v}")
except Exception as e:
    print("  [!] Could not access 'args' from the Ultralytics model.")

# Extra: inspect using PyTorch
print("\nDirect inspection using PyTorch:")
try:
    raw_model = torch.load(model_path, map_location='cpu')
    print("Keys found in the PyTorch dictionary:")
    print(list(raw_model.keys()))

    if 'train_args' in raw_model:
        print("\nTraining arguments ('train_args') found:")
        for k, v in raw_model['train_args'].items():
            print(f"  - {k}: {v}")
    else:
        print("  [!] No 'train_args' found in the model.")

except Exception as e:
    print(f"[ERROR] Failed to load model with PyTorch: {e}")
