# `sources/main.cpp` – PID control with joystick & shared‑memory mask

This file wires the entire **JetRacer autonomous lane‑keeping loop**:

* **Input** – receives a pre‑segmented **binary lane mask** from another process via POSIX shared memory (`/dev/shm/mask_shared`).
* **Perception** – calls `jetracer::pid::PIDexecute` to extract the lateral error and compute the steering angle.
* **Actuation** – sends the filtered command to the steering servo using `JetRacer::smooth_steering()`.
* **Human interaction** – throttle remains under manual joystick control; only steering is corrected.

> **Note**: Requires a **producer process** (Python or C++) that writes a 128 × 128 mask into the same shared memory area (byte 0 = flag, bytes 1‑16384 = image).

---

## Execution flow

```mermaid
graph TD
    A[Program start] --> B["Init JetRacer (I2C 0x40 &sol; 0x60)"]
    B --> C["Map shared memory \"mask_shared\""]
    C --> D[while true]
    D -->|flag==0| D
    D -->|flag==1| E[PIDexecute with mask]
    E --> F["smooth_steering(angle)"]
    F --> G[flag = 0]
    G --> D
```

---

## Shared‑memory layout

| Offset | Size (bytes) | Purpose                                           |
| -----: | -----------: | ------------------------------------------------- |
|    `0` |          `1` | **flag** – 1 ⇒ new image ready / 0 ⇒ processed    |
|    `1` |      `16384` | **mask** – 128 × 128, 8‑bit grayscale (lane mask) |

---

## Key steps in `main()`

1. Print banner.
2. Instantiate `JetRacer` with I2C addresses `0x40` (servo) and `0x60` (motor). Store pointer for safe stop.
3. Register `signal_handler()` for **SIGINT**; ensures `stop()` on Ctrl + C.
4. Open and map the POSIX shared memory `mask_shared`.
5. Create a zero‑copy **`cv::Mat mask`** that wraps the mapped region.
6. **Infinite loop**:

   * Wait until `flag == 1` (new mask).
   * Generate sequential filename `frame_XXXX.jpg` for overlay.
   * Run `PIDexecute(mask.clone(), filename)` – `clone` used because the pipeline draws on the image.
   * Call `smooth_steering(angle, 5)` to soften steps.
   * Reset `flag` to 0 to signal “mask processed”.
   * Break on **Esc** key.
7. Call `stop()`, print “Finishing.”, exit.

---

## Signals & error handling

* **Ctrl + C** triggers `signal_handler()`, which stops the JetRacer immediately.
* Exceptions are caught; error message printed and `stop()` called.
* Robust checks on `shm_open` / `mmap`; failure → `stderr` + exit code 1.

---

## Build & run

```bash

# Terminal 2: compile
make

# Terminal 1: producer writing masks to shared memory
python3 scripts/camera_yolo_to_shm.py

# Terminal 2: this C++ binary
./bin//jetracer_pid_controler
```

**Dependencies**

* OpenCV ≥ 4.5
* JetRacer libraries (`jetracer::control`, `jetracer::vision`, `jetracer::pid`)
* POSIX shared memory, mmap, signals (Linux)

---

## References

* POSIX Shared Memory (`shm_open`, `mmap`)
