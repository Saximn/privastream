from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid
from dotenv import load_dotenv
import os
import secrets
from typing import Dict, Any, Optional
import time

load_dotenv()

from src.core.config import web_config
from src.core.logging import logger
from src.web.backend import auth

# Constants
DEFAULT_MEDIASOUP_URL = 'http://localhost:3001'
ROOM_ID_LENGTH = 8
MAX_VOTE_BUFFER_SIZE = 3
DEFAULT_ROOM_TTL_SECONDS = 6 * 3600


def require_secret_key() -> str:
    """Return the configured SECRET_KEY, failing loudly in production.

    A missing key silently falls back to None in Flask, which weakens session
    signing. Require it in production; in development generate an ephemeral key
    (sessions won't survive a restart, which is fine for local dev).
    """
    key = os.getenv('SECRET_KEY')
    if key:
        return key
    if os.getenv('FLASK_ENV') == 'production':
        raise RuntimeError('SECRET_KEY environment variable must be set in production')
    logger.warning('SECRET_KEY not set; using an ephemeral development key')
    return secrets.token_hex(32)


def allowed_origins() -> list:
    """CORS allowlist from CORS_ALLOWED_ORIGINS (comma-separated).

    Defaults to the localhost frontend for development. Never wildcard.
    """
    raw = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:3000')
    return [o.strip() for o in raw.split(',') if o.strip()]

class RoomManager:
    """Manages room state and operations"""
    
    def __init__(self, room_ttl_seconds: Optional[int] = None):
        self.rooms: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.room_ttl_seconds = room_ttl_seconds or int(
            os.getenv("ROOM_TTL_SECONDS", str(DEFAULT_ROOM_TTL_SECONDS))
        )

    def _now(self) -> int:
        return int(time.time())

    def cleanup_expired_rooms(self) -> int:
        """Drop inactive rooms and associated user state."""
        now = self._now()
        expired = [
            room_id
            for room_id, room in self.rooms.items()
            if room.get("expires_at", 0) <= now
        ]
        for room_id in expired:
            self._delete_room(room_id)
        return len(expired)

    def _delete_room(self, room_id: str):
        room = self.rooms.pop(room_id, None)
        if not room:
            return
        sids = {room.get("host"), *room.get("viewers", [])}
        for sid in filter(None, sids):
            user = self.users.get(sid)
            if user and user.get("room") == room_id:
                user["room"] = None
                user["role"] = None

    def touch_room(self, room_id: str):
        if room_id in self.rooms:
            now = self._now()
            self.rooms[room_id]["last_seen_at"] = now
            self.rooms[room_id]["expires_at"] = now + self.room_ttl_seconds
    
    def create_room(self, host_sid: str) -> str:
        """Create a new room with host"""
        self.cleanup_expired_rooms()
        # Opaque, high-entropy id (the old uuid4()[:8] was guessable/brute-forceable).
        room_id = secrets.token_urlsafe(9)
        now = self._now()
        self.rooms[room_id] = {
            'host': host_sid,
            'viewers': [],
            'sfu_ready': False,
            'created_at': now,
            'last_seen_at': now,
            'expires_at': now + self.room_ttl_seconds,
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
        self.cleanup_expired_rooms()
        if room_id not in self.rooms:
            return False
        if viewer_sid not in self.rooms[room_id]['viewers']:
            self.rooms[room_id]['viewers'].append(viewer_sid)
        self.touch_room(room_id)
        return True
    
    def leave_room(self, room_id: str, user_sid: str, is_host: bool = False):
        """Remove user from room"""
        if room_id not in self.rooms:
            return
        
        if is_host:
            self._delete_room(room_id)
        else:
            self.rooms[room_id]['viewers'] = [
                v for v in self.rooms[room_id]['viewers'] if v != user_sid
            ]
            self.touch_room(room_id)
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get room information"""
        self.cleanup_expired_rooms()
        return self.rooms.get(room_id)
    
    def set_sfu_status(self, room_id: str, status: bool):
        """Update SFU streaming status"""
        if room_id in self.rooms:
            self.rooms[room_id]['sfu_ready'] = status
            self.touch_room(room_id)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = require_secret_key()
CORS(app, origins=allowed_origins())
socketio = SocketIO(
    app,
    path='/backend/socket.io',
    cors_allowed_origins=allowed_origins(),
    async_mode=os.getenv('SOCKETIO_ASYNC_MODE', 'threading'),
    logger=os.getenv('SOCKETIO_LOGGER', 'false').lower() == 'true',
    engineio_logger=os.getenv('SOCKETIO_ENGINEIO_LOGGER', 'false').lower() == 'true',
)

# Configuration
MEDIASOUP_SERVER_URL = os.getenv('MEDIASOUP_SERVER_URL', DEFAULT_MEDIASOUP_URL)

# Initialize room manager
room_manager = RoomManager()

@app.route('/health')
def health():
    """Health check endpoint"""
    expired = room_manager.cleanup_expired_rooms()
    return {
        'status': 'healthy',
        'rooms': len(room_manager.rooms),
        'users': len(room_manager.users),
        'expired_rooms_cleaned': expired,
    }

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
        
        # Issue a signed host token the client forwards to the SFU / detection
        # service. No-op when SECRET_KEY is unset (local dev) so room creation
        # keeps working; enforcement is gated by REQUIRE_AUTH on each service.
        token = None
        try:
            if os.getenv('SECRET_KEY'):
                token = auth.issue_room_token(room_id, role='host')
        except Exception as exc:
            logger.warning(f'Could not issue room token: {exc}')

        emit('room_created', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL, 'token': token})
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
        
        token = None
        try:
            if os.getenv('SECRET_KEY'):
                token = auth.issue_room_token(room_id, role='viewer')
        except Exception as exc:
            logger.warning(f'Could not issue viewer room token: {exc}')

        emit('joined_room', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL, 'token': token})
        
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
    """Return the configured module-level Flask app and Socket.IO server."""
    return app, socketio

if __name__ == '__main__':
    try:
        logger.info('Starting Privastream backend server')
        socketio.run(app, host=web_config.FLASK_HOST, port=web_config.FLASK_PORT, debug=web_config.FLASK_DEBUG, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f'Failed to start server: {e}')
        raise
