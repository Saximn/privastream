from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

load_dotenv()

# Add privastream to path  
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from privastream.core.config import web_config
from privastream.core.logging import logger

# Constants
DEFAULT_MEDIASOUP_URL = 'http://localhost:3001'
ROOM_ID_LENGTH = 8
MAX_VOTE_BUFFER_SIZE = 3

class RoomManager:
    """Manages room state and operations"""
    
    def __init__(self):
        self.rooms: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
    
    def create_room(self, host_sid: str) -> str:
        """Create a new room with host"""
        room_id = str(uuid.uuid4())[:ROOM_ID_LENGTH]
        self.rooms[room_id] = {
            'host': host_sid,
            'viewers': [],
            'sfu_ready': False
        }
        return room_id
    
    def add_user(self, sid: str, user_id: str, role: Optional[str] = None, room: Optional[str] = None):
        """Add user to tracking"""
        self.users[sid] = {
            'id': user_id,
            'role': role,
            'room': room
        }
    
    def remove_user(self, sid: str) -> Optional[Dict[str, Any]]:
        """Remove user and return their info"""
        return self.users.pop(sid, None)
    
    def join_room(self, room_id: str, viewer_sid: str) -> bool:
        """Add viewer to room"""
        if room_id not in self.rooms:
            return False
        self.rooms[room_id]['viewers'].append(viewer_sid)
        return True
    
    def leave_room(self, room_id: str, user_sid: str, is_host: bool = False):
        """Remove user from room"""
        if room_id not in self.rooms:
            return
        
        if is_host:
            del self.rooms[room_id]
        else:
            self.rooms[room_id]['viewers'] = [
                v for v in self.rooms[room_id]['viewers'] if v != user_sid
            ]
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get room information"""
        return self.rooms.get(room_id)
    
    def set_sfu_status(self, room_id: str, status: bool):
        """Update SFU streaming status"""
        if room_id in self.rooms:
            self.rooms[room_id]['sfu_ready'] = status

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
CORS(app, origins="*")
socketio = SocketIO(app, path='/backend/socket.io', cors_allowed_origins="*", 
                   async_mode='threading', logger=True, engineio_logger=True)

# Configuration
MEDIASOUP_SERVER_URL = os.getenv('MEDIASOUP_SERVER_URL', DEFAULT_MEDIASOUP_URL)

# Initialize room manager
room_manager = RoomManager()

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy'}

@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    try:
        user_id = str(uuid.uuid4())
        room_manager.add_user(request.sid, user_id)
        emit('connected', {'userId': user_id})
        logger.info(f'User {user_id} connected (SID: {request.sid})')
    except Exception as e:
        logger.error(f'Error handling connect: {e}')
        emit('error', {'message': 'Connection failed'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        user = room_manager.users.get(request.sid)
        if user and user['room']:
            room_id = user['room']
            leave_room(room_id)
            
            room_info = room_manager.get_room_info(room_id)
            if room_info:
                if user['role'] == 'host':
                    socketio.emit('host_disconnected', room=room_id)
                    room_manager.leave_room(room_id, request.sid, is_host=True)
                    logger.info(f'Host disconnected, room {room_id} closed')
                else:
                    room_manager.leave_room(room_id, request.sid, is_host=False)
                    socketio.emit('viewer_left', {'userId': user['id']}, room=room_id)
                    logger.info(f'Viewer {user["id"]} left room {room_id}')
        
        room_manager.remove_user(request.sid)
    except Exception as e:
        logger.error(f'Error handling disconnect: {e}')

@socketio.on('create_room')
def handle_create_room():
    """Handle room creation request"""
    try:
        room_id = room_manager.create_room(request.sid)
        join_room(room_id)
        
        # Update user info
        user = room_manager.users.get(request.sid)
        if user:
            user['role'] = 'host'
            user['room'] = room_id
        
        emit('room_created', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL})
        logger.info(f'Room {room_id} created with SFU support (Host: {request.sid})')
    except Exception as e:
        logger.error(f'Error creating room: {e}')
        emit('error', {'message': 'Failed to create room'})

@socketio.on('join_room')
def handle_join_room(data):
    """Handle room join request"""
    try:
        room_id = data.get('roomId')
        if not room_id:
            emit('error', {'message': 'Room ID required'})
            return
        
        if not room_manager.join_room(room_id, request.sid):
            emit('error', {'message': 'Room not found'})
            return
        
        join_room(room_id)
        
        # Update user info
        user = room_manager.users.get(request.sid)
        if user:
            user['role'] = 'viewer'
            user['room'] = room_id
        
        emit('joined_room', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL})
        
        # Check if streaming is already active
        room_info = room_manager.get_room_info(room_id)
        if room_info and room_info.get('sfu_ready', False):
            emit('streaming_started', {'roomId': room_id})
            logger.info(f'Notified new viewer about active streaming in room {room_id}')
        
        # Notify other users
        viewer_count = len(room_info['viewers']) if room_info else 0
        socketio.emit('viewer_joined', {
            'userId': user['id'] if user else 'unknown', 
            'viewerCount': viewer_count
        }, room=room_id)
        logger.info(f'User joined room {room_id} with SFU support')
        
    except Exception as e:
        logger.error(f'Error joining room: {e}')
        emit('error', {'message': 'Failed to join room'})

# SFU-related event handlers (WebRTC signaling now handled by Mediasoup server)
@socketio.on('sfu_streaming_started')
def handle_sfu_streaming_started(data):
    """Handle SFU streaming start notification"""
    try:
        user = room_manager.users.get(request.sid)
        if not user or user['role'] != 'host':
            emit('error', {'message': 'Only hosts can start streaming'})
            return
        
        room_id = user['room']
        if room_id:
            room_manager.set_sfu_status(room_id, True)
            socketio.emit('streaming_started', {'roomId': room_id}, room=room_id)
            logger.info(f'SFU streaming started for room {room_id}')
    except Exception as e:
        logger.error(f'Error starting SFU streaming: {e}')
        emit('error', {'message': 'Failed to start streaming'})

@socketio.on('sfu_streaming_stopped')
def handle_sfu_streaming_stopped(data):
    """Handle SFU streaming stop notification"""
    try:
        user = room_manager.users.get(request.sid)
        if not user or user['role'] != 'host':
            emit('error', {'message': 'Only hosts can stop streaming'})
            return
        
        room_id = user['room']
        if room_id:
            room_manager.set_sfu_status(room_id, False)
            socketio.emit('streaming_stopped', {'roomId': room_id}, room=room_id)
            logger.info(f'SFU streaming stopped for room {room_id}')
    except Exception as e:
        logger.error(f'Error stopping SFU streaming: {e}')
        emit('error', {'message': 'Failed to stop streaming'})

@socketio.on('get_room_info')
def handle_get_room_info(data):
    """Handle room information request"""
    try:
        room_id = data.get('roomId')
        if not room_id:
            emit('error', {'message': 'Room ID required'})
            return
        
        room_info = room_manager.get_room_info(room_id)
        if room_info:
            emit('room_info', {
                'exists': True,
                'viewerCount': len(room_info['viewers'])
            })
        else:
            emit('room_info', {'exists': False})
    except Exception as e:
        logger.error(f'Error getting room info: {e}')
        emit('error', {'message': 'Failed to get room info'})

def create_app():
    """Application factory for Flask app."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    CORS(app, origins="*")
    
    socketio = SocketIO(app, path='/backend/socket.io', cors_allowed_origins="*", 
                       async_mode='threading', logger=True, engineio_logger=True)
    
    return app, socketio

if __name__ == '__main__':
    try:
        logger.info('Starting Privastream backend server')
        socketio.run(app, host=web_config.FLASK_HOST, port=web_config.FLASK_PORT, debug=web_config.FLASK_DEBUG, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f'Failed to start server: {e}')
        raise
