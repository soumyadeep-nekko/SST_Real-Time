# 🛡️ SST Vision – Real-Time RTSP Camera Monitoring with YOLOv8 + Flask

This project uses [Ultralytics YOLOv8](https://docs.ultralytics.com/) for real-time object tracking and alert generation over an **RTSP video stream**, wrapped in a Flask web app.

It tracks people, vehicles, and mobile phone usage with visual and textual alerts, and displays live inference logs.

---

## 🚀 Features

- 📹 **Live RTSP Video Feed** (via OpenCV)
- 🧠 **YOLOv8 Object Detection & ByteTrack Tracking**
- 📍 Detects:
  - People
  - Vehicles (car, bus, truck, motorcycle, etc.)
  - Cell Phone usage by people
- 🔁 **Dwell Time Monitoring**:
  - Alerts when a vehicle is idle > 60s
  - Warning after 45s
- 🧾 **Live Logs**:
  - Event alerts (in red/yellow/green)
  - Inference summaries per frame
- 🌐 **Web Interface** with Streamed MJPEG and EventSource logs

---

## 📦 Requirements

Ensure you are using Python 3.8+ and install the following:

```bash
pip install flask opencv-python ultralytics
