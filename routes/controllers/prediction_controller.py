import json
from wsgiref.simple_server import WSGIRequestHandler

from django.http import JsonResponse
from apps.prediction_bot.prediction_bot import PredictionBot
from channels.generic.websocket import AsyncWebsocketConsumer
from django.core.handlers.asgi import ASGIRequest
prediction_bot = PredictionBot()


class PredictionController:
    @staticmethod
    async def predict(data: ASGIRequest):
        try:
            symbol = data.GET['symbol']

            if symbol:
                predicitons = await prediction_bot.predict(symbol=symbol)
                return JsonResponse({'predicitons': predicitons})
            else:
                # Symbol not found in the query parameters
                return JsonResponse({'error': 'Symbol not provided'})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    @staticmethod
    async def predict_subscribe(data, websocket: AsyncWebsocketConsumer):
        try:
            symbol = data.get('symbol')

            def callback(predictions): return websocket.send(
                json.dumps({"predictions": predictions}))

            await prediction_bot.predict_subscribe(symbol=symbol, callback=callback)

        except Exception as e:
            await websocket.send({'error': str(e)})
