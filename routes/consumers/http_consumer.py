# consumers.py
from django.http import JsonResponse
from channels.generic.http import AsyncHttpConsumer


class HTTPConsumer(AsyncHttpConsumer):
    async def handle(self, body):
        await self.send_response(200, 'success')
