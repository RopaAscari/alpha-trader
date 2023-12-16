import webbrowser
import tensorflow as tf
from apps.prediction_bot.settings import EPOCHS


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, client):
        self.client = client

    async def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / EPOCHS
        print(f'on progress {progress}')
        await self.client.send(progress)

# Define a custom TensorBoard callback


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_train_begin(self, logs=None):
        # Launch TensorBoard when training starts
        # webbrowser.open("http://localhost:6006/")
        super().on_train_begin(logs)
