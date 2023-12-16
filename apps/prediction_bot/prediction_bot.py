import asyncio
import logging
import numpy as np
import pandas as pd
from modules.logger import AppLogger
from apps.neural_network import settings
from sklearn.preprocessing import MinMaxScaler
from modules.services.deriv_service import DerivService
from apps.neural_network.neural_network import NeuralNetwork

# Set up the logger
logger = AppLogger('prediction_bot_logger', logging.DEBUG)
neural_network = NeuralNetwork()


class PredictionBot:
    def __init__(self):
        self.tick_count = 0
        self.tick_list = []

    async def predict_subscribe(self, symbol: str, callback: None):
        await self.get_predicitions_subscribe(symbol=symbol, callback=callback)

    async def predict(self, symbol: str):
        return await self.get_predicitions_nonsubscribe(symbol=symbol)

    async def get_predicitions_nonsubscribe(self, symbol,):
        symbol_data = await DerivService.get_instance().get_history_for_symbol(symbol=symbol, start=1, end="latest", count=15)

        pip_size = symbol_data["pip_size"]

        dataframe = pd.DataFrame({
            "times": symbol_data["history"]["times"],
            "prices": symbol_data["history"]["prices"],
        })

        predictions = self.generate_predictions(dataframe, pip_size)

        return predictions

    async def get_predicitions_subscribe(self, symbol, callback):
        symbol_data = await DerivService.get_instance().get_history_for_symbol(symbol=symbol, start=1, end="latest", count=15)

        ticks = await DerivService.get_instance().get_ticks_for_symbol(symbol=symbol)

        dataframe = pd.DataFrame({
            "prices": symbol_data["history"]["prices"],
            "times": symbol_data["history"]["times"]
        })

        pip_size = symbol_data["pip_size"]

        preditions = self.generate_predictions(dataframe, pip_size)

        await callback(preditions)

        ticks.subscribe(on_next=lambda tick: asyncio.create_task(
            self.handle_tick(tick, pip_size, callback)))

        # Define a function to handle ticks
    async def handle_tick(self, tick, pip_size, callback):

        # Increment the tick_count
        self.tick_count += 1

        # Append the tick to the list
        self.tick_list.append(tick)

        # Check if we have received 15 ticks
        if self.tick_count == 15:
            # Create a DataFrame with the collected ticks
            subscription_dataframe = pd.DataFrame({
                "prices": [tick['tick']['quote'] for tick in self.tick_list],
                "times": [tick['tick']['epoch'] for tick in self.tick_list]
            })

            # Reset the tick_count and tick_list
            self.tick_count = 0
            self.tick_list = []

            # Call the generate prediction method with the DataFrame
            predicitions = self.generate_predictions(
                subscription_dataframe, pip_size)

            await callback(predicitions)

    def generate_predictions(self, dataframe, pip_size):

        scaler_pred = MinMaxScaler()
        # Extract the latest timestamp from the DataFrame
        latest_time = dataframe['times'].iloc[-1]

        values = dataframe[settings.FEATURE_COLUMNS].values

        values = values.reshape(-1, len(settings.FEATURE_COLUMNS))

        scaled_values = scaler_pred.fit_transform(values)

        # Increment it by 2 for each prediction
        time_step = 2

        # Prepare input for prediction
        x_test_latest_batch = self.prepare_input_for_prediction(scaled_values)

        # Make predictions
        y_pred_scaled = neural_network.network.predict(x_test_latest_batch)

        # Inverse transform the predictions
        y_pred_unscaled = scaler_pred.inverse_transform(y_pred_scaled)

        # Flatten the unscaled predictions
        predictions_values = y_pred_unscaled.flatten().tolist()

        # Separate prices and times
        prices = predictions_values  # It's now a 1D array of prices

        times = []

        for price in prices:
            times.append(int(latest_time))
            latest_time += time_step

        prices = [round(price, pip_size) for price in prices]

        # Create a list of dictionaries
        results = []
        for price, time in zip(prices, times):
            result = {"price": price, "time": time}
            results.append(result)

        return results

    def prepare_input_for_prediction(self, scaled_values):

        # Define the desired input shape
        # Adjust this to match your input sequence length
        sequence_length = settings.INPUT_SEQUENCE
        num_features = len(settings.FEATURE_COLUMNS)
        # Ensure that scaled_values has the correct shape (15, 2)
        if scaled_values.shape != (sequence_length, num_features):
            raise ValueError(
                f"scaled_values should have shape ({sequence_length}, {num_features})")

        # Create a new array for the input with shape (1, sequence_length, num_features)
        input_data = np.zeros((1, sequence_length, num_features))

        # Fill the input_data with the last 15 time steps from scaled_values
        input_data[0, :, :] = scaled_values[-sequence_length:]

        return input_data
