# PPE Detection & Face Recognition System

Real-time safety monitoring system that combines Personal Protective Equipment (PPE) detection with facial recognition for construction sites and industrial facilities.

## Features

- **PPE Detection**: Real-time detection of safety equipment compliance using YOLO v11
- **Face Recognition**: Person identification using MediaPipe + DeepFace with FaceNet embeddings
- **Web Interface**: Live video stream with real-time detection status dashboard
- **API Integration**: RESTful endpoints for external system integration
- **Snapshot Capture**: Save annotated images with detection results
- **Database Management**: Automatic face embedding generation and caching
- **GPU Support**: CUDA acceleration for improved performance

## Dataset

This project uses the **Construction Site Safety Image Dataset** from Kaggle, specifically designed for PPE detection in construction environments.

### Dataset Details
- **Total Classes**: 10
- **Focus**: Distinguishes between wearing and not wearing safety equipment
- **Environment**: Construction site scenarios with various lighting and angles

### Class Mapping
```
0: Hardhat          5: Person
1: Mask             6: Safety Cone
2: NO-Hardhat       7: Safety Vest
3: NO-Mask          8: Machinery
4: NO-Safety Vest   9: Vehicle
```

## Model Performance

### YOLOv11 Large (PPE Detection)
- **Precision**: 0.9363
- **Recall**: 0.8207
- **mAP@50**: 0.8798
- **mAP@50-95**: 0.6131

### Face Recognition
- **Model**: FaceNet via DeepFace
- **Detection**: MediaPipe BlazeFace
- **Similarity Method**: Cosine similarity
- **Threshold**: 0.7 (configurable)

## Implementation

### Architecture
```
Camera Input → Frame Processing → Dual Detection Pipeline
                                       ↓
           PPE Detection (YOLO v11) + Face Recognition (MediaPipe + DeepFace)
                                       ↓
           Result Annotation → Flask Backend → Web Interface + API
```

### Tech Stack
- **Backend**: Flask, YOLO v11, MediaPipe, DeepFace, OpenCV, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, jQuery
- **Database**: Pickle-based face embedding cache
- **Models**: Pre-trained YOLO v11 Large + FaceNet embeddings

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU acceleration)
Webcam or camera source
```

### Dependencies
```bash
pip install flask ultralytics opencv-python mediapipe deepface
pip install scikit-learn numpy torch torchvision
```

### Setup Process

1. **Download Model**
   ```bash
   # Place yolov11_large.pt in project directory
   ```

2. **Create Face Database Structure**
   ```
   face_data/
   ├── john_doe/
   │   ├── john1.jpg
   │   ├── john2.jpg
   │   └── john3.jpg
   ├── jane_smith/
   │   ├── jane1.jpg
   │   └── jane2.jpg
   └── embeddings.pkl  # Auto-generated
   ```

3. **Configure Paths in app.py**
   ```python
   PPE_MODEL_PATH = "path/to/yolov11_large.pt"
   FACE_DB_PATH = "path/to/face_data"
   ```

4. **Run Application**
   ```bash
   python app.py
   ```
   Access: `http://localhost:5000`

## API Endpoints

### Detection Status
```bash
GET /api/ppe_status
```
Returns current PPE compliance and identified persons
```json
{
    "hardhat": true,
    "mask": false,
    "safety_vest": true,
    "person": true,
    "faces_detected": "2 face(s) detected",
    "person_1": "John Doe (0.85)",
    "person_2": "Jane Smith (0.92)"
}
```

### Detailed Status
```bash
GET /api/detection_status
```
Comprehensive detection data with timestamps

### Snapshot Capture
```bash
POST /api/snapshot
```
Captures annotated image with all detections

## Configuration Options

### Camera Settings
```python
camera = cv2.VideoCapture(1)  # Change camera index (0, 1, 2...)
```

### Detection Thresholds
```python
# PPE Detection
conf_threshold = 0.5          # YOLO confidence threshold
imgsz = 640                   # Input image size

# Face Recognition  
similarity_threshold = 0.7     # Face matching threshold
min_detection_confidence = 0.5 # Face detection confidence
```

### Performance Optimization
```python
# GPU Usage (automatic detection)
if torch.cuda.is_available():
    model.to("cuda")
```

## Use Cases

- **Construction Site Monitoring**: Real-time PPE compliance tracking
- **Industrial Safety**: Worker identification and equipment verification
- **Access Control**: Authorized personnel verification with safety check
- **Compliance Documentation**: Automated safety audit trails
- **Incident Investigation**: Historical snapshots with person identification

## Upgrade from Previous Version

### New Capabilities
- ✅ **Face Recognition System**: Complete person identification pipeline
- ✅ **Enhanced Database**: Face embedding management and caching
- ✅ **Advanced API**: Multiple endpoints for different data needs
- ✅ **Snapshot Integration**: Captures both PPE and face detection results
- ✅ **Performance Optimization**: GPU acceleration and efficient processing

### Previous Version
- Basic PPE detection only
- Simple web interface
- Limited API functionality

## Future Enhancements

- **Multi-camera Support**: Simultaneous monitoring from multiple sources
- **Database Integration**: PostgreSQL/MySQL for production deployment
- **Advanced Analytics**: Historical compliance reporting and trends
- **Mobile Application**: Remote monitoring capabilities
- **Alert System**: Real-time notifications for safety violations
- **Extended PPE Classes**: Additional safety equipment detection

