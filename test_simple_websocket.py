#!/usr/bin/env python3
"""
Simple WebSocket Connection Test
===============================

Quick test to verify WebSocket connectivity issues with the web server.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_secret'

# Very simple SocketIO configuration
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.5/socket.io.js"></script>
</head>
<body>
    <h1>WebSocket Connection Test</h1>
    <div id="status">Disconnected</div>
    <button id="testBtn">Send Test Message</button>
    <div id="log"></div>

    <script>
        const socket = io();
        const status = document.getElementById('status');
        const log = document.getElementById('log');
        
        function addLog(message) {
            log.innerHTML += message + '<br>';
            console.log(message);
        }
        
        socket.on('connect', function() {
            status.textContent = 'Connected!';
            status.style.color = 'green';
            addLog('✅ Connected to server');
        });
        
        socket.on('disconnect', function() {
            status.textContent = 'Disconnected';
            status.style.color = 'red';
            addLog('❌ Disconnected from server');
        });
        
        socket.on('test_response', function(data) {
            addLog('📥 Received: ' + data.message);
        });
        
        document.getElementById('testBtn').addEventListener('click', function() {
            addLog('📤 Sending test message...');
            socket.emit('test_message', {message: 'Hello from client!'});
        });
    </script>
</body>
</html>
    '''

@socketio.on('connect')
def handle_connect():
    print("🔗 Client connected!")
    emit('test_response', {'message': 'Welcome! Connection successful'})

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Client disconnected")

@socketio.on('test_message')
def handle_test_message(data):
    print(f"📥 Received test message: {data}")
    emit('test_response', {'message': f'Server received: {data["message"]}'})

if __name__ == '__main__':
    print("🚀 Starting simple WebSocket test server...")
    print("📱 Open browser to: http://localhost:5000")
    print("⚡ Press Ctrl+C to stop")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
