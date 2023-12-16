import json

from modules.keras.callbacks import ProgressCallback
from apps.neural_network.neural_network import NeuralNetwork
from apps.prediction_bot.prediction_bot import PredictionBot
from channels.generic.websocket import AsyncWebsocketConsumer

nueral_network = NeuralNetwork()


class TrainingController:
    @staticmethod
    async def train_subscribe(data, websocket: AsyncWebsocketConsumer):
        try:
            symbol = data.get('symbol')

            progress_callback = ProgressCallback(websocket)

            def callback(progress): return websocket.send(
                json.dumps({"progress": progress}))

            await nueral_network.train(symbol=symbol, progress_callback=callback)

        except Exception as e:
            await websocket.send({'error': str(e)})
