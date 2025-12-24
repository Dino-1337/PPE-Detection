
import cv2
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import os

from .face_database import FaceDatabase
from .violation_logger import ViolationLogger
import config

logger = logging.getLogger(__name__)


class PPEFaceDetectionSystem:
    """Main detection system for PPE and face recognition"""
    
    def __init__(self, ppe_model_path: str, face_db_path: str, 
                 camera_index: int = 0, face_recognition_interval: int = 30,
                 violation_logger: ViolationLogger = None):
        self.ppe_model = YOLO(ppe_model_path)
        if torch.cuda.is_available():
            self.ppe_model.to("cuda")
            logger.info("PPE model loaded on GPU")
        else:
            logger.info("PPE model loaded on CPU")
        
        self.CLASS_MAP = {
            0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
            4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
            7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
        }
        
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,  # 0 for short-range (<2m), 1 for full-range
            min_detection_confidence=0.5
        )
        
        self.face_db = FaceDatabase(face_db_path)
        
        self.violation_logger = violation_logger
        
        self.camera = cv2.VideoCapture(camera_index)
        self.camera_active = True
        
        # Performance: Frame skipping for face recognition
        self.face_recognition_interval = face_recognition_interval
        self.frame_counter = 0
        self.cached_face_results = []
        
        self.latest_detections = {
            'hardhat': False,
            'mask': False,
            'safety_vest': False,
            'person': False,
            'faces_detected': 'No faces detected',
            'face_identities': []
        }
    
    def detect_ppe(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect PPE in frame"""
        results = self.ppe_model.predict(frame, imgsz=640, conf=config.PPE_CONFIDENCE_THRESHOLD, verbose=False)

        detections = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
        
        ppe_status = {
            'hardhat': 0 in detections,
            'mask': 1 in detections,
            'safety_vest': 7 in detections,
            'person': 5 in detections
        }
        
        annotated_frame = results[0].plot() if results[0].boxes is not None else frame
        return annotated_frame, ppe_status
    
    def detect_and_recognize_faces(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Detect faces and recognize identities"""
        face_results = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(frame_rgb)
        
        if results.detections:
            h, w, _ = frame.shape
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                try:
                    # Crop face
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Get face embedding (pass image directly, no disk I/O)
                        embedding = DeepFace.represent(
                            img_path=face_crop,  # Pass numpy array directly
                            model_name='Facenet',
                            enforce_detection=False
                        )[0]['embedding']
                        
                        # Identify face
                        identity, confidence = self.face_db.identify_face(np.array(embedding))
                        
                        face_results.append({
                            'bbox': (x1, y1, x2, y2),
                            'identity': identity,
                            'confidence': confidence
                        })
                            
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
        
        return face_results, frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with both PPE and face detection"""
        # PPE Detection (runs every frame)
        ppe_frame, ppe_status = self.detect_ppe(frame)
        
        # Face Detection and Recognition (optimized with frame skipping)
        self.frame_counter += 1
        if self.frame_counter % self.face_recognition_interval == 0:
            # Run face recognition every N frames
            face_results, _ = self.detect_and_recognize_faces(frame)
            self.cached_face_results = face_results
            
            # Check and log violations (only when face recognition runs)
            if self.violation_logger:
                self.violation_logger.check_and_log_violations(ppe_status, face_results, frame)
        else:
            # Use cached results for intermediate frames
            face_results = self.cached_face_results
        
        # Update cache for API - format for HTML compatibility
        self.latest_detections.update(ppe_status)
        
        # Format face results for HTML
        if face_results:
            faces_text = f"{len(face_results)} face(s) detected"
            self.latest_detections['faces_detected'] = faces_text
            
            # Add individual person entries that HTML looks for
            for i, face in enumerate(face_results):
                person_key = f"person_{i+1}"
                identity = face['identity']
                confidence = face['confidence']
                self.latest_detections[person_key] = f"{identity} ({confidence:.2f})"
        else:
            self.latest_detections['faces_detected'] = 'No faces detected'
            # Remove any existing person entries
            keys_to_remove = [k for k in self.latest_detections.keys() if k.startswith('person_')]
            for key in keys_to_remove:
                del self.latest_detections[key]
        
        # Store raw face results for other uses
        self.latest_detections['face_identities'] = face_results
        
        # Annotate frame with face recognition results
        final_frame = ppe_frame.copy()
        
        for face in face_results:
            x1, y1, x2, y2 = face['bbox']
            identity = face['identity']
            confidence = face['confidence']
            
            # Draw face bounding box (different color from PPE)
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw identity label
            label = f"{identity}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(final_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(final_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return final_frame
    
    def gen_frames(self):
        """Generate video frames for Flask streaming"""
        while True:
            if self.camera_active and self.camera.isOpened():
                success, frame = self.camera.read()
                if not success:
                    break
                
                processed_frame = self.process_frame(frame)
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def take_snapshot(self) -> Dict:
        """Take snapshot with all detections"""
        if not self.camera_active or not self.camera.isOpened():
            return {'success': False, 'message': 'Camera not active'}
        
        ret, frame = self.camera.read()
        if not ret:
            return {'success': False, 'message': 'Failed to capture frame'}
        
        processed_frame = self.process_frame(frame)
        
        # Save snapshot
        snapshots_dir = 'static/snapshots'
        os.makedirs(snapshots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ppe_face_snapshot_{timestamp}.jpg"
        filepath = os.path.join(snapshots_dir, filename)
        
        cv2.imwrite(filepath, processed_frame)
        
        return {
            'success': True,
            'message': f'Snapshot saved: {filename}',
            'filepath': filepath
        }
