#!/usr/bin/env python3
"""
Face Enrollment Server using InsightFace Buffalo_S
Provides live face detection and enrollment capabilities for the Mediasoup server
"""

import os
import sys
import base64
import json
import numpy as np
import cv2
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional

# InsightFace imports
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Install with: pip install insightface")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FaceEnrollmentSystem:
    def __init__(self):
        self.face_app = None
        self.room_embeddings = {}  # roomId -> {'embeddings': [embeddings], 'metadata': {...}}
        self.detection_model = None
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize InsightFace Buffalo_S model"""
        if not INSIGHTFACE_AVAILABLE:
            logger.error("InsightFace not available. Please install: pip install insightface")
            return False
            
        try:
            logger.info("Initializing InsightFace Buffalo_S model...")
            
            # Initialize face analysis with Buffalo_S model
            self.face_app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare the model
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("✅ InsightFace Buffalo_S initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize InsightFace: {e}")
            return False
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in an image and return bounding boxes with confidence"""
        if self.face_app is None:
            return []
            
        try:
            start_time = time.time()
            
            # Analyze faces
            faces = self.face_app.get(image)
            
            detection_time = time.time() - start_time
            
            detected_faces = []
            for face in faces:
                # Extract bounding box and confidence
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                confidence = float(face.det_score)
                
                detected_faces.append({
                    'bbox': bbox.tolist(),
                    'confidence': confidence
                })
            
            logger.info(f"Detected {len(detected_faces)} faces in {detection_time:.3f}s")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def extract_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract face embeddings from an image"""
        if self.face_app is None:
            return []
            
        try:
            faces = self.face_app.get(image)
            embeddings = []
            
            for face in faces:
                # Extract normalized embedding
                embedding = face.normed_embedding
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return []
    
    def enroll_face(self, room_id: str, frames: List[str]) -> Dict:
        """Enroll a face using multiple frames to compute average embedding"""
        try:
            logger.info(f"Starting face enrollment for room: {room_id}")
            logger.info(f"Processing {len(frames)} frames")
            
            all_embeddings = []
            valid_frames = 0
            
            for i, frame_data in enumerate(frames):
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        logger.warning(f"Failed to decode frame {i}")
                        continue
                    
                    # Extract embeddings from this frame
                    embeddings = self.extract_embeddings(image)
                    
                    if embeddings:
                        all_embeddings.extend(embeddings)
                        valid_frames += 1
                        logger.info(f"Frame {i}: extracted {len(embeddings)} face embedding(s)")
                    else:
                        logger.warning(f"Frame {i}: no faces detected")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    continue
            
            if not all_embeddings:
                return {
                    'success': False,
                    'message': f'No face embeddings extracted from {len(frames)} frames',
                    'enrollment_complete': False
                }
            
            # Compute average embedding
            if len(all_embeddings) > 1:
                # Stack embeddings and compute mean
                embeddings_array = np.stack(all_embeddings)
                average_embedding = np.mean(embeddings_array, axis=0)
                # Normalize the average embedding
                average_embedding = average_embedding / np.linalg.norm(average_embedding)
            else:
                average_embedding = all_embeddings[0]
            
            # Store in room embeddings
            self.room_embeddings[room_id] = {
                'embedding': average_embedding,
                'metadata': {
                    'enrollment_time': datetime.now().isoformat(),
                    'frames_processed': len(frames),
                    'valid_frames': valid_frames,
                    'embeddings_count': len(all_embeddings)
                }
            }
            
            logger.info(f"✅ Face enrollment complete for room {room_id}")
            logger.info(f"   Processed: {valid_frames}/{len(frames)} frames")
            logger.info(f"   Embeddings: {len(all_embeddings)} -> 1 average")
            
            return {
                'success': True,
                'message': f'Face enrolled successfully from {valid_frames} frames',
                'enrollment_complete': True,
                'metadata': self.room_embeddings[room_id]['metadata']
            }
            
        except Exception as e:
            logger.error(f"Face enrollment error: {e}")
            return {
                'success': False,
                'message': f'Enrollment failed: {str(e)}',
                'enrollment_complete': False
            }
    
    def get_room_embedding(self, room_id: str) -> Optional[np.ndarray]:
        """Get the enrolled face embedding for a room"""
        if room_id in self.room_embeddings:
            return self.room_embeddings[room_id]['embedding']
        return None
    
    def cleanup_room(self, room_id: str):
        """Remove enrollment data for a room"""
        if room_id in self.room_embeddings:
            del self.room_embeddings[room_id]
            logger.info(f"Cleaned up enrollment data for room: {room_id}")

# Initialize the face enrollment system
enrollment_system = FaceEnrollmentSystem()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = {
        'buffalo_s': enrollment_system.face_app is not None
    }
    
    return jsonify({
        'status': 'healthy',
        'service': 'Face Enrollment Server (InsightFace Buffalo_S)',
        'models_loaded': models_loaded,
        'insightface_available': INSIGHTFACE_AVAILABLE,
        'active_rooms': len(enrollment_system.room_embeddings),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/face-detection', methods=['POST'])
def face_detection():
    """Live face detection endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'frame_data' not in data:
            return jsonify({
                'success': False,
                'error': 'No frame data provided'
            }), 400
        
        frame_data = data['frame_data']
        room_id = data.get('room_id', 'unknown')
        detect_only = data.get('detect_only', False)
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode image'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image decode error: {str(e)}'
            }), 400
        
        # Detect faces
        start_time = time.time()
        faces_detected = enrollment_system.detect_faces(image)
        detection_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'faces_detected': faces_detected,
            'detection_time': detection_time,
            'room_id': room_id,
            'detect_only': detect_only
        })
        
    except Exception as e:
        logger.error(f"Face detection endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}'
        }), 500

@app.route('/api/face-enrollment', methods=['POST'])
def face_enrollment():
    """Face enrollment endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'frames' not in data or 'room_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing frames or room_id'
            }), 400
        
        frames = data['frames']
        room_id = data['room_id']
        
        if not isinstance(frames, list) or len(frames) == 0:
            return jsonify({
                'success': False,
                'error': 'No frames provided for enrollment'
            }), 400
        
        # Perform enrollment
        result = enrollment_system.enroll_face(room_id, frames)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Face enrollment endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Enrollment failed: {str(e)}',
            'enrollment_complete': False
        }), 500

@app.route('/api/room-status/<room_id>', methods=['GET'])
def get_room_status(room_id: str):
    """Get enrollment status for a room"""
    try:
        embedding = enrollment_system.get_room_embedding(room_id)
        
        if embedding is not None:
            metadata = enrollment_system.room_embeddings[room_id]['metadata']
            return jsonify({
                'enrolled': True,
                'room_id': room_id,
                'metadata': metadata
            })
        else:
            return jsonify({
                'enrolled': False,
                'room_id': room_id
            })
            
    except Exception as e:
        return jsonify({
            'error': f'Status check failed: {str(e)}'
        }), 500

@app.route('/api/cleanup-room/<room_id>', methods=['DELETE'])
def cleanup_room(room_id: str):
    """Clean up enrollment data for a room"""
    try:
        enrollment_system.cleanup_room(room_id)
        return jsonify({
            'success': True,
            'message': f'Room {room_id} cleaned up'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cleanup failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Face Enrollment Server with InsightFace Buffalo_S")
    
    if not INSIGHTFACE_AVAILABLE:
        logger.error("❌ InsightFace not available. Please install: pip install insightface")
        logger.error("   You may also need: pip install onnxruntime-gpu")
        sys.exit(1)
    
    if enrollment_system.face_app is None:
        logger.error("❌ Failed to initialize face detection models")
        sys.exit(1)
    
    # Start the Flask server
    port = int(os.environ.get('PORT', 5003))
    logger.info(f"🌟 Face Enrollment Server starting on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )