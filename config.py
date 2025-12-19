# Configuration for PPE Detection & Face Recognition System

import os

# Model Paths
PPE_MODEL_PATH = r"Z:/AI PROJECTS/PPE Detection/LivePPE/yolov11_large.pt"
FACE_DB_PATH = r"Z:/AI PROJECTS/PPE Detection/Datasets/face_data"

# Camera Settings
CAMERA_INDEX = 0  # 0 for default camera, 1 for external webcam
CAMERA_ACTIVE = True

# Performance Settings
FACE_RECOGNITION_INTERVAL = 30  # Run every N frames (30 = ~1 second at 30 FPS)

# Detection Thresholds
PPE_CONFIDENCE_THRESHOLD = 0.5
PPE_IMAGE_SIZE = 640
FACE_DETECTION_CONFIDENCE = 0.5
FACE_DETECTION_MODEL = 0  # 0 for short-range (<2m), 1 for full-range
FACE_SIMILARITY_THRESHOLD = 0.7
FACE_RECOGNITION_MODEL = 'Facenet'

# Flask Settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# PPE Class Mapping
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
