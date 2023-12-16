# consumers.py
import json
from routes.routers.websocket_router import websocket_router
from channels.generic.websocket import AsyncWebsocketConsumer


class WebsocketConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):

        data = json.loads(text_data)

        route = data.get('operation')

        if route in websocket_router:
            await websocket_router[route](data, self)
        else:
            await self.handle_default(data)

    async def handle_default(self, data):
        await self.send_error_message("Unknown message type")

    async def send_error_message(self, error_message):
        await self.send(json.dumps({"type": "training.error", "error": error_message}))
