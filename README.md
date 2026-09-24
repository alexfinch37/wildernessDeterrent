# wildernessDeterrent 2026

Wilderness Deterrent is a humane backyard wildlife deterrent. A Raspberry Pi 4 with an HQ camera runs a custom YOLOv8 model trained on ~1,365 images across three classes (raccoon, squirrel, person). When a raccoon or squirrel is detected, and no person is in frame, the Pi signals an Arduino Uno over USB serial to spin up dual RS2205 brushless flywheels and launch a biodegradable packing peanut. A Flask web server streams live MJPEG video and exposes a manual fire button and a status endpoint.

Stack: Python, YOLOv8 (Ultralytics), Picamera2, Flask, Arduino, 3S LiPo with BLHeli_S ESCs, and 3D-printed mounts designed in CadQuery and Fusion 360.

## Getting Started

**Hardware:** Raspberry Pi 4 (64-bit Bookworm), HQ Camera + 6mm CS lens, Arduino Uno, 2× RS2205 motors with 30A BLHeli_S ESCs, 3S LiPo

1. Flash the Arduino sketch in `/arduino` to the Uno and connect it to the Pi via USB (defaults to `/dev/ttyACM0`).
2. On the Pi, install dependencies:
```bash
   pip install -r requirements.txt
```
3. Start the server:
```bash
   python v8server.py
```
4. Open `http://<pi-ip>:5000` to view the live feed and manual fire control.

The trained model weights (`peopleYolov8.pt`) are included with three classes trained.
