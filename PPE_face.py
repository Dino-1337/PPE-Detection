from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
import torch
import mediapipe as mp
from deepface import DeepFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FaceDatabase:
    def __init__(self, face_data_path: str):
        self.face_data_path = face_data_path
        self.embeddings_cache = {}
        self.person_names = []
        self.embeddings_matrix = None
        self.load_face_database()
    
    def load_face_database(self):
        """Load and process face database"""
        try:
            embeddings_file = os.path.join(self.face_data_path, 'embeddings.pkl')
            
            if os.path.exists(embeddings_file):
                # Load cached embeddings
                with open(embeddings_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embeddings_cache = cache_data.get('embeddings', {})
                    self.person_names = cache_data.get('names', [])
                logger.info(f"Loaded cached embeddings for {len(self.person_names)} people")
            else:
                # Create embeddings from images
                self.create_embeddings_from_images()
                
            # Convert to matrix for faster similarity search
            if self.embeddings_cache:
                embeddings_list = []
                names_list = []
                for name, embedding in self.embeddings_cache.items():
                    embeddings_list.append(embedding)
                    names_list.append(name)
                
                self.embeddings_matrix = np.array(embeddings_list)
                self.person_names = names_list
                logger.info(f"Face database ready with {len(self.person_names)} identities")
            else:
                logger.warning("No face embeddings found!")
                
        except Exception as e:
            logger.error(f"Error loading face database: {e}")
            self.embeddings_matrix = np.array([])
            self.person_names = []
    
    def create_embeddings_from_images(self):
        """Create embeddings from face images in the database"""
        if not os.path.exists(self.face_data_path):
            logger.error(f"Face data path doesn't exist: {self.face_data_path}")
            return
            
        embeddings = {}
        
        # Iterate through person folders
        for person_name in os.listdir(self.face_data_path):
            person_path = os.path.join(self.face_data_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            person_embeddings = []
            
            # Process images for this person
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    
                    try:
                        # Extract embedding using DeepFace
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name='Facenet',  # Fast and accurate
                            enforce_detection=False
                        )[0]['embedding']
                        person_embeddings.append(embedding)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {img_path}: {e}")
            
            if person_embeddings:
                # Average multiple embeddings for same person
                avg_embedding = np.mean(person_embeddings, axis=0)
                embeddings[person_name] = avg_embedding
                logger.info(f"Created embedding for {person_name} from {len(person_embeddings)} images")
        
        self.embeddings_cache = embeddings
        
        # Save embeddings cache
        os.makedirs(self.face_data_path, exist_ok=True)
        with open(os.path.join(self.face_data_path, 'embeddings.pkl'), 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'names': list(embeddings.keys())
            }, f)
    
    def identify_face(self, face_embedding: np.ndarray, threshold: float = 0.7) -> Tuple[str, float]:
        """Identify face using cosine similarity"""
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return "Unknown", 0.0
        
        try:
            # Calculate cosine similarity
            similarities = cosine_similarity([face_embedding], self.embeddings_matrix)[0]
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity >= threshold:
                return self.person_names[best_match_idx], float(best_similarity)
            else:
                return "Unknown", float(best_similarity)
                
        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return "Unknown", 0.0


class PPEFaceDetectionSystem:
    def __init__(self, ppe_model_path: str, face_db_path: str):
        # Initialize PPE Detection
        self.ppe_model = YOLO(ppe_model_path)
        if torch.cuda.is_available():
            self.ppe_model.to("cuda")
            logger.info("PPE model loaded on GPU")
        else:
            logger.info("PPE model loaded on CPU")
        
        # PPE Class mapping
        self.CLASS_MAP = {
            0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
            4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
            7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
        }
        
        # Initialize Face Detection (BlazeFace via MediaPipe)
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Initialize Face Database
        self.face_db = FaceDatabase(face_db_path)
        
        # Camera
        self.camera = cv2.VideoCapture(1)
        self.camera_active = True
        
        # Detection results cache for API
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
        results = self.ppe_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
        detections = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
        
        # Determine PPE status
        ppe_status = {
            'hardhat': 0 in detections,  # Hardhat detected
            'mask': 1 in detections,     # Mask detected
            'safety_vest': 7 in detections,  # Safety Vest detected
            'person': 5 in detections    # Person detected
        }
        
        # Get annotated frame
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
                        # Get face embedding
                        temp_path = "temp_face.jpg"
                        cv2.imwrite(temp_path, face_crop)
                        
                        embedding = DeepFace.represent(
                            img_path=temp_path,
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
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
        
        return face_results, frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with both PPE and face detection"""
        # PPE Detection
        ppe_frame, ppe_status = self.detect_ppe(frame)
        
        # Face Detection and Recognition
        face_results, _ = self.detect_and_recognize_faces(frame)
        
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


# Initialize the system
PPE_MODEL_PATH = r"Z:/AI PROJECTS/PPE Detection/LivePPE/yolov11_large.pt"
FACE_DB_PATH = r"Z:/AI PROJECTS/PPE Detection/Datasets/face_data"

detection_system = PPEFaceDetectionSystem(PPE_MODEL_PATH, FACE_DB_PATH)

# Flask routes
@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(detection_system.gen_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/ppe_status')
def ppe_status():
    """Return PPE and face detection status in format expected by camera.html"""
    return jsonify(detection_system.latest_detections)

@app.route('/api/detection_status')
def detection_status():
    """Alternative endpoint - returns detailed status"""
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
    app.run(debug=True, host='0.0.0.0', port=5000)