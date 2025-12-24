
import os
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class FaceDatabase:
    
    def __init__(self, face_data_path: str):
        self.face_data_path = face_data_path
        self.embeddings_cache = {}
        self.person_names = []
        self.embeddings_matrix = None
        self.load_face_database()
    
    def load_face_database(self):
        try:
            embeddings_file = os.path.join(self.face_data_path, 'embeddings.pkl')
            
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embeddings_cache = cache_data.get('embeddings', {})
                    self.person_names = cache_data.get('names', [])
                logger.info(f"Loaded cached embeddings for {len(self.person_names)} people")
            else:
                self.create_embeddings_from_images()
                
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
        if not os.path.exists(self.face_data_path):
            logger.error(f"Face data path doesn't exist: {self.face_data_path}")
            return
            
        embeddings = {}
        
        for person_name in os.listdir(self.face_data_path):
            person_path = os.path.join(self.face_data_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            person_embeddings = []
            
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    
                    try:
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name='Facenet',
                            enforce_detection=False
                        )[0]['embedding']
                        person_embeddings.append(embedding)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {img_path}: {e}")
            
            if person_embeddings:
                # Average multiple embeddings for robustness
                avg_embedding = np.mean(person_embeddings, axis=0)
                embeddings[person_name] = avg_embedding
                logger.info(f"Created embedding for {person_name} from {len(person_embeddings)} images")
        
        self.embeddings_cache = embeddings
        
        os.makedirs(self.face_data_path, exist_ok=True)
        with open(os.path.join(self.face_data_path, 'embeddings.pkl'), 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'names': list(embeddings.keys())
            }, f)
    
    def identify_face(self, face_embedding: np.ndarray, threshold: float = 0.7) -> Tuple[str, float]:
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return "Unknown", 0.0
        
        try:
            similarities = cosine_similarity([face_embedding], self.embeddings_matrix)[0]
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity >= threshold:
                return self.person_names[best_match_idx], float(best_similarity)
            else:
                return "Unknown", float(best_similarity)
                
        except Exception as e:
            logger.error(f"Error in face identification: {e}")
            return "Unknown", 0.0
