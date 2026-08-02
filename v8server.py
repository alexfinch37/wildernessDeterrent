import serial
import time
import threading
import cv2
from flask import Flask, Response, render_template_string, jsonify
from picamera2 import Picamera2
from ultralytics import YOLO


import werkzeug.serving
werkzeug.serving.WSGIRequestHandler.address_string = lambda self: self.client_address[0]


# --- Serial Setup ---
arduino = serial.Serial('/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_749373038363515040C0-if00', 9600, timeout=1)
time.sleep(2)
arduino_lock = threading.Lock()

# --- Model + Camera Setup ---
model = YOLO('peopleYolov8.pt')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

# Target classes — only auto-fire on these
TARGET_CLASSES = {'raccoon', 'squirrel'}
COOLDOWN = 5.0
last_fired = 0.0

# Auto-fire toggle (default: on)
auto_fire_enabled = True

# Shared state between the detection thread and the Flask routes
latest_jpeg = None
frame_lock = threading.Lock()
last_status = "idle"
status_lock = threading.Lock()

app = Flask(__name__)


def fire(reason="auto"):
    """Send the fire command to the Arduino. Thread-safe."""
    global last_fired, last_status
    with arduino_lock:
        arduino.write(b'1')
    last_fired = time.time()
    with status_lock:
        last_status = f"fired ({reason})"
    print(f"Firing! reason={reason}")


def detection_loop():
    """Runs continuously in the background: captures frames, runs YOLO,
    auto-fires on target classes, and keeps the latest annotated frame
    available for the MJPEG stream."""
    global latest_jpeg, last_status

    while True:
        frame = picam2.capture_array()
        results = model(frame, imgsz=320, conf=0.5, verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes
        names = results[0].names  # index -> class name mapping

        detected = [
            names[int(box.cls)] for box in boxes
            if names[int(box.cls)].lower() in TARGET_CLASSES
        ]
        should_fire = len(detected) > 0

        if should_fire and auto_fire_enabled:
            now = time.time()
            if now - last_fired > COOLDOWN:
                print(f"Target detected: {detected} — firing!")
                fire(reason=f"auto: {detected}")
            else:
                with status_lock:
                    last_status = f"detected {detected} (cooling down)"
        elif should_fire:
            with status_lock:
                last_status = f"detected {detected} (auto-fire off)"
        else:
            with status_lock:
                if not last_status.startswith("fired"):
                    last_status = "watching"

        # Encode annotated frame as JPEG for the web stream
        ok, buf = cv2.imencode('.jpg', annotated)
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()


def mjpeg_generator():
    while True:
        with frame_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)


PAGE = """
<!doctype html>
<html>
<head>
  <title>Raccoon Cannon</title>
  <style>
    body { font-family: sans-serif; background: #1a1a1a; color: #eee; text-align: center; margin: 0; padding: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    #status { margin: 1rem 0; font-size: 1.1rem; color: #9fe1cb; }
    img { border: 4px solid #333; border-radius: 8px; max-width: 90vw; }
    button {
      margin-top: 1.5rem;
      font-size: 1.4rem;
      padding: 0.8rem 2rem;
      background: #d85a30;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    button:active { background: #993c1d; }
    button:disabled { background: #555; cursor: not-allowed; }
    #autoBtn { background: #2e8b57; margin-left: 1rem; }
    #autoBtn.off { background: #555; }
  </style>
</head>
<body>
  <h1>Raccoon Cannon</h1>
  <div id="status">status: loading...</div>
  <img src="{{ url_for('video_feed') }}">
    <img id="feed" src="/snapshot" alt="Live feed">
<script>
  const img = document.getElementById('feed');
  setInterval(() => {
    img.src = '/snapshot?t=' + Date.now();  // cache-buster
  }, 200);  // ~5 frames/sec, adjust to taste
</script>

  <br>
  <button id="fireBtn" onclick="manualFire()">MANUAL FIRE</button>
  <button id="autoBtn" onclick="toggleAuto()">AUTO FIRE: ON</button>

  <script>
    async function manualFire() {
      const btn = document.getElementById('fireBtn');
      btn.disabled = true;
      btn.innerText = 'FIRING...';
      try {
        await fetch('/fire', { method: 'POST' });
      } catch (e) {
        console.error(e);
      }
      setTimeout(() => {
        btn.disabled = false;
        btn.innerText = 'MANUAL FIRE';
      }, 1000);
    }

    function renderAuto(on) {
      const btn = document.getElementById('autoBtn');
      btn.innerText = 'AUTO FIRE: ' + (on ? 'ON' : 'OFF');
      btn.classList.toggle('off', !on);
    }

    async function toggleAuto() {
      try {
        const res = await fetch('/autofire', { method: 'POST' });
        const data = await res.json();
        renderAuto(data.auto_fire);
      } catch (e) {
        console.error(e);
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').innerText = 'status: ' + data.status;
        renderAuto(data.auto_fire);
      } catch (e) {}
    }
    setInterval(pollStatus, 1000);
    pollStatus();
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fire', methods=['POST'])
def manual_fire():
    fire(reason="manual")
    return jsonify({"ok": True})


@app.route('/autofire', methods=['POST'])
def toggle_autofire():
    global auto_fire_enabled
    auto_fire_enabled = not auto_fire_enabled
    return jsonify({"auto_fire": auto_fire_enabled})


@app.route('/status')
def status():
    with status_lock:
        return jsonify({"status": last_status, "auto_fire": auto_fire_enabled})

@app.route('/snapshot')
def snapshot():
    with frame_lock:
        frame = latest_jpeg
    if frame is None:
        return Response(status=503)
    return Response(frame, mimetype='image/jpeg')



if __name__ == '__main__':
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        arduino.close()
        picam2.stop()
