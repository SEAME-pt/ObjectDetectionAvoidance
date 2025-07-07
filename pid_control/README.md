# PID Control + JetRacer Vision

C++ implementation of lane detection, PID control, and hardware interface for the JetRacer car, integrated with a Python pipeline that writes camera segmentation masks to shared memory.

> **Requirements**
> • CMake ≥ 3.18 • OpenCV ≥ 4.5 • SDL2 • Linux I2C (i2c-dev) • Python ≥ 3.9 (for camera_yolo_to_shm.py)

---

## Repository Structure

apps/                # example executables
  └─ main.cpp        # PID with joystick + shared memory
includes/jetracer/   # public headers (*.hpp)
sources/             # C++ implementations (*.cpp)
models/              # YOLO / LaneNet model (.pt)
scripts/             # Python utilities
docs/                # Markdown documentation + diagrams
Makefile             # default target: make && make run

---

## Main Modules

| Module                   | Description                                  | Docs                                                                         |
| ------------------------ | -------------------------------------------- | ---------------------------------------------------------------------------- |
| `computer_vision`        | Lane extraction, center calculation, overlay | [`docs/docs_computer_vision_module.md`](docs/docs_computer_vision_module.md) |
| `pid_controller`         | Simplified PID with anti-wind-up             | [`docs/docs_pid_module.md`](docs/docs_pid_module.md)                         |
| `control` (`JetRacer`)   | PWM, servo, motors, joystick handling        | [`docs/docs_control_module.md`](docs/docs_control_module.md)                 |
| `hardware` (`I2CDevice`) | RAII wrapper for `/dev/i2c-X`                | [`docs/docs_hardware_module.md`](docs/docs_hardware_module.md)               |

---

## Execution

# Terminal 1: producer writes lane masks to shared memory
python3 scripts/camera_yolo_to_shm.py

# Terminal 2: run the C++ controller
./bin/jetracer_pid_controler

---

## Execution Flow (Simplified)

graph TD
    P1["Python – YOLO mask"] --> M1["/dev/shm/mask_shared"]
    M1 --> C1["apps/main.cpp"]
    C1 --> V1["computer_vision"]
    V1 --> PID1["pid_controller"]
    PID1 --> J1["JetRacer::smooth_steering"]



