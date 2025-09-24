# Emergency Stop System

## Overview

The emergency stop system has been implemented to automatically detect obstacles in the danger zone and trigger an immediate stop when occupancy exceeds 25% of the area.

## Implemented Components

### 1. Occupancy Calculation Function (`calculateDangerZoneOccupancy`)

**Location:** `sources/computer_vision.cpp`

**Functionality:**
- Calculates the percentage of area occupied by obstacles in the danger zone
- Creates a danger zone mask based on detected lane curves
- Identifies white pixels (obstacles) within the danger zone
- Returns an occupancy percentage (0.0 to 1.0)

**Parameters:**
- `mask`: Binary image mask
- `left_curve`: Left lane points
- `right_curve`: Right lane points
- `displacement_cm`: Lane width in centimeters
- `scale`: Pixel/cm conversion scale

### 2. Emergency Stop Function (`emergency_stop`)

**Location:** `sources/jetracer.cpp`

**Functionality:**

- Maintains speed 0 until the danger zone is clear
- Centers the steering
- Updates all state variables
- Stops speed tests if active
- Displays detailed log messages

### 3. Integration in Main Loop

**Location:** `apps/main.cpp`

**Functionality:**
- Detects lane curves in real-time
- Calculates danger zone occupancy each frame
- Triggers emergency stop if occupancy > 25%
- Waits 500ms after stop before continuing (optimized)

### 4. Visual Indicators in Stream

**Functionality:**
- Shows danger zone occupancy percentage
- Dynamic colors:
  - Green: < 15% (safe)
  - Yellow: 15-25% (attention)
  - Red: > 25% (emergency)
- Displays "EMERGENCY STOP!" when activated

## Operation Flow

```
1. Camera frame capture
2. Binary mask processing
3. Lane curve detection
4. Danger zone calculation
5. Obstacle identification in zone
6. Occupancy percentage calculation
7. Threshold verification (25%)
8. If exceeded: Trigger emergency stop
9. Update visual indicators
10. Continue to next frame
```

## Configuration

### Emergency Threshold
- **Value:** 25% (0.25)
- **Location:** `apps/main.cpp` - constant `EMERGENCY_THRESHOLD`
- **Adjustable:** Yes, modify the constant

### Visual Color Thresholds
- **Green:** < 15% (safe)
- **Yellow:** 15-25% (attention)
- **Red:** > 25% (emergency)

## Safety

### Safety Features:

4. **Centering:** Steering is automatically centered
5. **Clean State:** All variables are reset
6. **Detailed Logs:** Complete action recording
7. **Recovery Time:** 500ms pause after stop (optimized)

### False Positive Prevention:
- Uses robust lane detection
- Considers only the lower half of the image
- Applies morphological filters to reduce noise
- Requires significant occupancy (25%) to activate
