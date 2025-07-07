# `computer_vision` module – `jetracer::vision`

The `computer_vision` module, located in the `jetracer::vision` namespace, implements functions for **lane detection and analysis** in images using **OpenCV**.

It targets **autonomous vehicle navigation** scenarios, where the system must quickly identify the track’s centre and compute the **lateral error** for downstream controllers (e.g. PID, MPC).

---

## Constants

These constants define the default dimensions of the processed images and their total pixel count:

```cpp
constexpr int WIDTH  = 128;
constexpr int HEIGHT = 128;
constexpr int SIZE   = WIDTH * HEIGHT;
```

**Typical use‑cases**

* Fixed‑size buffer allocation
* Input‑image validation

---

## Public API

### 1. `float getXAtY(float y, float y0, float x0, float vx, float vy);`

**Computes the X coordinate corresponding to a given Y on a straight line defined by the point `(x0, y0)` and direction vector `(vx, vy)`.**

Handy when you need to find where a line crosses a chosen horizontal scanline in the image.

| Parameter | Description                             |
| --------- | --------------------------------------- |
| `y`       | Target Y coordinate                     |
| `y0`      | Y coordinate of the line’s reference pt |
| `x0`      | X coordinate of the line’s reference pt |
| `vx`      | X component of the direction vector     |
| `vy`      | Y component of the direction vector     |

**Returns** the X coordinate at the requested Y.

---

### 2. `bool extractLanePoints(...)`

```cpp
bool extractLanePoints(const cv::Mat &frame,
                       float            image_center,
                       float           &y_ref,
                       std::vector<cv::Point> &left_point,
                       std::vector<cv::Point> &right_point);
```

**Extracts representative points of the left and right lane markings inside a Region of Interest (ROI).**

Contour detection and line fitting are used to locate each lane.

| Parameter      | Description                                               |
| -------------- | --------------------------------------------------------- |
| `frame`        | Binary image (lanes already segmented)                    |
| `image_center` | Horizontal centre of the frame                            |
| `y_ref`        | <ins>output</ins> – reference Y where points were sampled |
| `left_point`   | <ins>output</ins> – vector of points for the left lane    |
| `right_point`  | <ins>output</ins> – vector of points for the right lane   |

**Returns** `true` if at least one lane is detected, `false` otherwise.

**Key techniques**

* Image segmentation
* `cv::findContours`
* `cv::fitLine`
* Slope analysis to separate left/right lanes

---

### 3. `float calculateTrackCenter(...)`

```cpp
float calculateTrackCenter(const std::vector<cv::Point> &left,
                           const std::vector<cv::Point> &right,
                           float   y_ref,
                           float   displacement_cm,
                           float   scale,
                           cv::Mat &frame);
```

**Calculates the track centre on the reference line `y_ref` from the left and right lane points.**

If only one lane is available, the centre is estimated using the physical lane width and a pixel‑to‑cm scale.

| Parameter         | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `left`            | Points belonging to the left lane                    |
| `right`           | Points belonging to the right lane                   |
| `y_ref`           | Y coordinate of the reference scanline               |
| `displacement_cm` | Physical track width (cm)                            |
| `scale`           | Pixels‑per‑centimetre conversion factor              |
| `frame`           | Image where visual markers (circles, etc.) are drawn |

**Returns** the X coordinate of the track centre at `y_ref`, or `‑1.0f` when no lane is detected.

**Key techniques**

* Line fitting with `cv::fitLine`
* Mid‑point estimation
* OpenCV visualisation (`cv::circle`)

---

### 4. `void draw_overlay(...)`

```cpp
void draw_overlay(cv::Mat            &frame,
                  float               error,
                  float               pid,
                  const std::string  &file_name,
                  const std::string  &txt_lane,
                  float               image_center,
                  float               center_track,
                  float               y_ref);
```

**Renders an informative overlay on the image for debugging purposes.**

It draws:

* Reference lines
* Calculated track centre
* Lateral error value
* PID correction value
* Auxiliary text labels

| Parameter      | Description                            |
| -------------- | -------------------------------------- |
| `frame`        | Original image to be annotated         |
| `error`        | Lateral error (pixels or degrees)      |
| `pid`          | PID correction value                   |
| `file_name`    | Frame identifier or filename           |
| `txt_lane`     | Text related to lane status            |
| `image_center` | Horizontal centre of the image         |
| `center_track` | Computed track centre                  |
| `y_ref`        | Reference Y coordinate for the overlay |

**Returns** nothing (`void`).

**Key techniques**

* `cv::line`, `cv::circle`, `cv::putText`
* Real‑time visual inspection overlays

---

## Flow

graph TD
    %% ===== TOP-LEVEL PIPELINE =====
    subgraph Pipeline  [Vision pipeline]
        A["Binary mask (128×128)"] --> B["extractLanePoints"]
        B -->|"lanes detected"| C["calculateTrackCenter"]
        B -->|"no lane"| Z["Return −1"]
        C --> D["draw_overlay (optional)"]
        C --> E["centre X + y_ref"]
    end

    %% ====== extractLanePoints DETAILS ======
    subgraph B_details  ["Inside extractLanePoints()"]
        B1["Loop ROIs (row 3 … 6)"] --> B2["findContours"]
        B2 --> B3{"contour.size() ≥ 5?"}
        B3 -- no --> B1
        B3 -- yes --> B4["fitLine(contour)"]
        B4 --> B5["slope & distance ↦ classify • left / right"]
        B5 --> B6{"better than previous?"}
        B6 -- no --> B1
        B6 -- yes --> B7["store centre_left / centre_right"]
        B7 --> B1
        B1 --> B8{"found at least one lane?"}
        B8 -- no --> Z
        B8 -- yes --> B9["return left/right pts & y_ref"]
    end

    %% ====== calculateTrackCenter DETAILS ======
    subgraph C_details  ["Inside calculateTrackCenter()"]
        C1{"both lanes present?"}
        C1 -- yes --> C2["fitLine (left) & fitLine (right)"]
        C2 --> C3["x_left / x_right @ y_ref"]
        C3 --> C7["centre = (xL+xR)/2"]
        C1 -- only right --> C4["x_right @ y_ref"]
        C4 --> C5["centre = xR − disp_cm/scale"]
        C1 -- only left --> C6["x_left @ y_ref"]
        C6 --> C5
        C5 --> C7
        C7 --> C8["draw circles  ●"]
        C8 --> E
    end

---

## Core Concepts

* **Image Processing**

  * Binary segmentation
  * Contour extraction (`cv::findContours`)
  * Robust line fitting (`cv::fitLine`)
* **Analytical Geometry**

  * Parametric line equation
  * Midpoint calculation between lanes
* **Visualisation**

  * OpenCV drawing primitives for debugging
* **Robotics / Autonomous Vehicles**

  * Extraction of control variables (lateral error, centre position)

---

## Notes

* The module assumes the input image is **pre‑segmented** (lane markings highlighted).
* Using `cv::fitLine` improves robustness against noise and partial occlusions.
* `draw_overlay` eases both off‑line log inspection and real‑time visual validation.

---

## References

* [OpenCV Official Documentation](https://docs.opencv.org/)
* Analytic Geometry for Computer Vision (various resources)
