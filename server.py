import asyncio
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from livekit import api

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://ai-agent-5lweuwtz.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APIzFsc98f2USae")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "KI3qL7rcEddClhWegWrBMnwibXNrkuCeXyZ4gyAEQU5")

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/token':
            grants = api.VideoGrants(room_join=True, room='voice-agent', can_publish=True, can_subscribe=True)
            token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            token.with_identity('user').with_grants(grants)
            jwt = token.to_jwt()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"token": jwt}).encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass

os.chdir('/workspace/agent')
print("Server running on http://0.0.0.0:8090")
HTTPServer(('0.0.0.0', 8090), Handler).serve_forever()