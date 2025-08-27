## 📑 Services Documentation - Raspberry Pi

This document describes the services configured on the **Raspberry Pi** to automate and manage local processes.  


## 📌 Overview
The services listed here were created to keep scripts and applications running automatically after system boot, ensuring that critical tasks run continuously.


## 🔧 Created Services

### 1. **Service**
- **Description:** 
    **autonomy-mqtt-cloud.service** → Publishes car autonomy data to the cloud via MQTT.

    **autonomy-mqtt-local.service** → Calculates and publishes car autonomy data locally via MQTT.

    **battery-mqtt-cloud.service** → Publishes battery status to the cloud via MQTT.

    **can-manager.service** → Manages the CAN interface (activate, monitor, deactivate).

    **github-runner.service** → Runs GitHub Actions self-hosted runner on the Raspberry Pi.

    **speed-mqtt-cloud.service** → Publishes car speed data to the cloud via MQTT.

    **speed-mqtt-local.service** → Publishes car speed data locally via MQTT.

    **temperature-mqtt-cloud.service** → Publishes Raspberry Pi temperature data from local to the cloud via MQTT.
- **File location:**  
  `/etc/systemd/system`
- **Usage commands:**

  ```bash
  # Start the service
  sudo systemctl start service_name

  # Stop the service
  sudo systemctl stop service_name

  # Restart the service
  sudo systemctl restart service_name

  # Check status
  sudo systemctl status service_name
