# JetRacer Joystick Controls Reference

This document provides a complete reference of joystick control mappings used in the JetRacer system, including analog axes, buttons, and their respective system identifications.

## Overview

The JetRacer system uses a standard USB joystick for manual vehicle control. Controls are mapped through the Linux input system (`/dev/input/js0`) and include:

- **Analog axes**: For steering and speed control
- **Digital buttons**: For specific functions and emergency
- **D-pad**: For menu navigation

## Analog Axes Mapping

### D-Pad (Directional)
| Direction | Axis | Value | Description |
|-----------|------|-------|-------------|
| Up | 7 | < 0 | Upward movement |
| Down | 7 | > 0 | Downward movement |
| Left | 6 | < 0 | Leftward movement |
| Right | 6 | > 0 | Rightward movement |

### Right Stick (Right Thumbstick)
| Direction | Axis | Value | Description |
|-----------|------|-------|-------------|
| Up | 3 | < 0 | Positive vertical movement |
| Down | 3 | > 0 | Negative vertical movement |
| Left | 2 | < 0 | Negative horizontal movement |
| Right | 2 | > 0 | Positive horizontal movement |

### Left Stick (Left Thumbstick)
| Direction | Axis | Value | Description |
|-----------|------|-------|-------------|
| Up | 1 | < 0 | Positive vertical movement |
| Down | 1 | > 0 | Negative vertical movement |
| Left | 0 | < 0 | Negative horizontal movement |
| Right | 0 | > 0 | Positive horizontal movement |

## Button Mapping

### Shoulder Buttons
| Button | ID | State | Description |
|--------|----|-------|-------------|
| R1 | 7 | ON | Right upper button |
| R2 | 9 | ON | Right trigger |
| L1 | 6 | ON | Left upper button |
| L2 | 8 | ON | Left trigger |

### Action Buttons
| Button | ID | State | Description |
|--------|----|-------|-------------|
| Y | 4 | ON | Y button (yellow) |
| X | 3 | ON | X button (blue) |
| B | 1 | ON | B button (red) |
| A | 0 | ON | A button (green) |

## Usage in JetRacer System

### Main Controls
- **Left Stick (Axis 0)**: Vehicle steering control
- **Right Stick (Axis 2)**: Speed/acceleration control
- **A Button**: Start/stop system
- **B Button**: Emergency mode
- **X Button**: Toggle test mode
- **Y Button**: Reset settings

### Input Values
- **Negative values (< 0)**: Movement in one direction
- **Positive values (> 0)**: Movement in opposite direction
- **Zero value (0)**: Neutral/center position

## Testing and Verification

### Test Commands
```bash
# Test connected joystick
jstest /dev/input/js0

# Check input devices
ls /dev/input/

# Monitor events in real time
cat /dev/input/js0
```

### Online Tools
- **Hardware Tester**: https://hardwaretester.com/gamepad
- **Gamepad Tester**: https://gamepad-tester.com/

## Troubleshooting

### Common Issues
1. **Joystick not detected**: Check USB connection and drivers
2. **Controls not responding**: Check device access permissions
3. **Incorrect values**: Calibrate joystick using `jscal`

### Diagnostic Commands
```bash
# Check permissions
ls -la /dev/input/js0

# Test connectivity
sudo jstest /dev/input/js0

# Check events
sudo cat /dev/input/js0
```

## References

- **Linux Joystick API**: Official Linux kernel documentation
- **jstest**: Joystick testing tool
- **Hardware Tester**: https://hardwaretester.com/gamepad

---

**Last updated**: January 2025  
**Version**: 1.0
