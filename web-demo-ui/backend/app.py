from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
import uuid
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
SSL_ENABLED = os.getenv('SSL_ENABLED', 'false').lower() == 'true'
SSL_CERT_PATH = os.getenv('SSL_CERT_PATH', './ssl/cert.pem')
SSL_KEY_PATH = os.getenv('SSL_KEY_PATH', './ssl/key.pem')
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*')
SOCKETIO_CORS_ALLOWED_ORIGINS = os.getenv('SOCKETIO_CORS_ALLOWED_ORIGINS', '*')
SOCKETIO_LOGGER = os.getenv('SOCKETIO_LOGGER', 'false').lower() == 'true'
SOCKETIO_ENGINEIO_LOGGER = os.getenv('SOCKETIO_ENGINEIO_LOGGER', 'false').lower() == 'true'

# Service URLs
MEDIASOUP_SERVER_URL = os.getenv('MEDIASOUP_SERVER_URL', 'http://localhost:3001')
VIDEO_SERVICE_URL = os.getenv('VIDEO_SERVICE_URL', 'http://localhost:5001')
AUDIO_SERVICE_URL = os.getenv('AUDIO_SERVICE_URL', 'http://localhost:5002')
FACE_ENROLLMENT_API_URL = os.getenv('FACE_ENROLLMENT_API_URL', 'http://localhost:5003')

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Configure CORS
if CORS_ALLOWED_ORIGINS == '*':
    CORS(app, origins="*")
else:
    CORS(app, origins=CORS_ALLOWED_ORIGINS.split(','))

# Configure SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins=SOCKETIO_CORS_ALLOWED_ORIGINS, 
    async_mode='threading', 
    logger=SOCKETIO_LOGGER, 
    engineio_logger=SOCKETIO_ENGINEIO_LOGGER
)

print(f'[CONFIG] Flask server configured:')
print(f'  Host: {FLASK_HOST}')
print(f'  Port: {FLASK_PORT}')
print(f'  SSL Enabled: {SSL_ENABLED}')
print(f'  CORS Origins: {CORS_ALLOWED_ORIGINS}')
print(f'  MediaSoup URL: {MEDIASOUP_SERVER_URL}')

rooms = {}
users = {}

@app.route('/health')
def health():
    return {'status': 'healthy'}

@socketio.on('connect')
def handle_connect():
    user_id = str(uuid.uuid4())
    users[request.sid] = {
        'id': user_id,
        'role': None,
        'room': None
    }
    emit('connected', {'userId': user_id})
    print(f'[BACKEND] User {user_id} connected (SID: {request.sid})')

@socketio.on('disconnect')
def handle_disconnect():
    user = users.get(request.sid)
    if user and user['room']:
        room_id = user['room']
        leave_room(room_id)
        
        if room_id in rooms:
            if user['role'] == 'host':
                socketio.emit('host_disconnected', room=room_id)
                del rooms[room_id]
            else:
                rooms[room_id]['viewers'] = [v for v in rooms[room_id]['viewers'] if v != request.sid]
                socketio.emit('viewer_left', {'userId': user['id']}, room=room_id)
    
    if request.sid in users:
        del users[request.sid]
    print(f'User disconnected')

@socketio.on('create_room')
def handle_create_room():
    room_id = str(uuid.uuid4())[:8]
    join_room(room_id)
    
    rooms[room_id] = {
        'host': request.sid,
        'viewers': [],
        'sfu_ready': False
    }
    
    users[request.sid]['role'] = 'host'
    users[request.sid]['room'] = room_id
    
    emit('room_created', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL})
    print(f'[BACKEND] Room {room_id} created with SFU support (Host: {request.sid})')

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    
    if room_id not in rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    join_room(room_id)
    rooms[room_id]['viewers'].append(request.sid)
    users[request.sid]['role'] = 'viewer'
    users[request.sid]['room'] = room_id
    
    emit('joined_room', {'roomId': room_id, 'mediasoupUrl': MEDIASOUP_SERVER_URL})
    
    # If streaming is already active, notify the new viewer
    if rooms[room_id].get('sfu_ready', False):
        emit('streaming_started', {'roomId': room_id})
        print(f'[BACKEND] Notified new viewer about active streaming in room {room_id}')
    else:
        print(f'[BACKEND] Room {room_id} streaming not active (sfu_ready: {rooms[room_id].get("sfu_ready", False)})')
    
    socketio.emit('viewer_joined', {'userId': users[request.sid]['id'], 'viewerCount': len(rooms[room_id]['viewers'])}, room=room_id)
    print(f'User joined room {room_id} with SFU support')

# SFU-related event handlers (WebRTC signaling now handled by Mediasoup server)
@socketio.on('sfu_streaming_started')
def handle_sfu_streaming_started(data):
    room_id = users[request.sid]['room']
    if room_id and users[request.sid]['role'] == 'host':
        rooms[room_id]['sfu_ready'] = True
        socketio.emit('streaming_started', {'roomId': room_id}, room=room_id)
        print(f'SFU streaming started for room {room_id}')

@socketio.on('sfu_streaming_stopped')
def handle_sfu_streaming_stopped(data):
    room_id = users[request.sid]['room']
    if room_id and users[request.sid]['role'] == 'host':
        rooms[room_id]['sfu_ready'] = False
        socketio.emit('streaming_stopped', {'roomId': room_id}, room=room_id)
        print(f'SFU streaming stopped for room {room_id}')

@socketio.on('get_room_info')
def handle_get_room_info(data):
    room_id = data['roomId']
    if room_id in rooms:
        emit('room_info', {
            'exists': True,
            'viewerCount': len(rooms[room_id]['viewers'])
        })
    else:
        emit('room_info', {'exists': False})

if __name__ == '__main__':
    # SSL context configuration
    ssl_context = None
    if SSL_ENABLED:
        try:
            import ssl
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_KEY_PATH)
            print(f'[SSL] SSL enabled with cert: {SSL_CERT_PATH}')
        except Exception as e:
            print(f'[SSL] Failed to load SSL certificates: {e}')
            print('[SSL] Falling back to HTTP')
            ssl_context = None
    
    # Start server
    debug_mode = os.getenv('FLASK_ENV', 'production') != 'production'
    socketio.run(
        app, 
        host=FLASK_HOST, 
        port=FLASK_PORT, 
        debug=debug_mode,
        ssl_context=ssl_context
    )