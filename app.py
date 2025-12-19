"""
PPE Detection & Face Recognition System - Flask Application

Main entry point for the web application.
Run with: python app.py
"""

from flask import Flask, render_template, Response, jsonify
from datetime import datetime
import logging
import config
from models import FaceDatabase, PPEFaceDetectionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Initializing PPE Detection & Face Recognition System...")
detection_system = PPEFaceDetectionSystem(
    ppe_model_path=config.PPE_MODEL_PATH,
    face_db_path=config.FACE_DB_PATH,
    camera_index=config.CAMERA_INDEX,
    face_recognition_interval=config.FACE_RECOGNITION_INTERVAL,
    class_map=config.CLASS_MAP
)
logger.info("System initialized successfully!")


@app.route('/')
def index():
    """Main page with video feed and detection status"""
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        detection_system.gen_frames(), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/ppe_status')
def ppe_status():
    """Return PPE and face detection status"""
    return jsonify(detection_system.latest_detections)


@app.route('/api/detection_status')
def detection_status():
    """Return detailed status with timestamp"""
    return jsonify({
        'success': True,
        'ppe_status': {
            'hardhat': detection_system.latest_detections['hardhat'],
            'mask': detection_system.latest_detections['mask'],
            'safety_vest': detection_system.latest_detections['safety_vest'],
            'person': detection_system.latest_detections['person']
        },
        'face_identities': detection_system.latest_detections['face_identities'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/snapshot', methods=['POST'])
def take_snapshot():
    """Take snapshot with all detections"""
    result = detection_system.take_snapshot()
    return jsonify(result)


if __name__ == "__main__":
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(
        debug=config.FLASK_DEBUG, 
        host=config.FLASK_HOST, 
        port=config.FLASK_PORT
    )
