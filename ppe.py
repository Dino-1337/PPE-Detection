from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import torch
app = Flask(__name__)

# Load YOLO model
model = YOLO(r"Z:/AI PROJECTS/PPE Detection/LivePPE/yolov11_large.pt").to("cuda")  # Use "cpu" if no GPU

# Class map
CLASS_MAP = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}

# Camera
camera = cv2.VideoCapture(0)  # Change index if wrong
camera_active = True


@app.route('/')
def index():
    return render_template('cameraFeed.html')


def gen_frames():
    while True:
        if camera_active and camera.isOpened():
            success, frame = camera.read()
            if not success:
                break
            else:
                results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/ppe_status')
def ppe_status():
    """Return PPE detection status"""
    if not camera_active or not camera.isOpened():
        return jsonify({'success': False, 'message': 'Camera not active'}), 400

    ret, frame = camera.read()
    if not ret:
        return jsonify({'success': False, 'message': 'Failed to capture frame'}), 500

    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    detections = results[0].boxes.cls.tolist()

    # Counts
    hardhat = detections.count(0)
    mask = detections.count(1)
    vest = detections.count(7)
    persons = detections.count(5)

    return jsonify({
        'success': True,
        'hardhat': hardhat,
        'mask': mask,
        'safety_vest': vest,
        'persons': persons
    })


@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    """Take and save snapshot"""
    global camera

    if not camera_active or not camera.isOpened():
        return jsonify({'success': False, 'message': 'Camera not active'}), 400

    ret, frame = camera.read()
    if not ret:
        return jsonify({'success': False, 'message': 'Failed to capture frame'}), 500

    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    snapshots_dir = 'static/snapshots'
    os.makedirs(snapshots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ppe_snapshot_{timestamp}.jpg"
    filepath = os.path.join(snapshots_dir, filename)

    cv2.imwrite(filepath, annotated_frame)

    return jsonify({
        'success': True,
        'message': f'Snapshot saved: {filename}',
        'filepath': filepath
    })


if __name__ == "__main__":
    app.run(debug=True)