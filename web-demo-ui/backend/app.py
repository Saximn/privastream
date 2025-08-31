from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Configuration for Mediasoup server
MEDIASOUP_SERVER_URL = 'http://localhost:3001'

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
    print(f'🟢 [BACKEND] User {user_id} connected (SID: {request.sid})')

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
    print(f'🟢 [BACKEND] Room {room_id} created with SFU support (Host: {request.sid})')

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
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)