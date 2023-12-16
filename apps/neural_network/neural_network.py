import os
import time
import logging
import numpy as np
import pandas as pd
from .import settings
import tensorflow as tf
from modules.logger import AppLogger
from datetime import datetime, timedelta
from modules.utils import partition_dataset
from sklearn.preprocessing import MinMaxScaler
from modules.keras.callbacks import CustomTensorBoard
from modules.services.deriv_service import DerivService

logger = AppLogger('neural_network', logging.DEBUG)


class NeuralNetwork:
    network: tf.keras.Sequential
    deriv: DerivService.get_instance()

    @classmethod
    def initialize(cls):
        path = os.path.join(settings.NUERON_PATH,
                            settings.MODEL_PATH, settings.MODEL_NAME)
        logger.info(f'Path {path}...')
        if os.path.isfile(path):
            logger.info(f'Attempting to load path {path}...')
            loaded_model = tf.keras.models.load_model(path)

            # Check its architecture
            loaded_model.summary()
            logger.info(f'Neural network was successfully initialized!')
            cls.network = loaded_model  # Assign to the class attribute
        else:
            cls.create_neural_network()  # Use cls to call the class method

        tf.data.experimental.enable_debug_mode()
        return True

    @classmethod
    def create_neural_network(cls):
        logger.info(f'Attempting to create a neural network...')

        # Create required folders if they don't exist
        folders_to_create = [
            os.path.join(settings.NUERON_PATH, settings.MODEL_PATH,
                         settings.BASE_LOG_FOLDER),
            os.path.join(settings.NUERON_PATH, settings.MODEL_PATH,
                         settings.BASE_HISTORY_FOLDER),
            os.path.join(settings.NUERON_PATH, settings.MODEL_PATH,
                         settings.BASE_CHECKPOINT_FOLDER)
        ]

        for folder in folders_to_create:
            os.makedirs(folder, exist_ok=True)

            n_features = len(settings.FEATURE_COLUMNS)
            # input_layer = tf.keras.layers.Input(
            #     shape=(settings.INPUT_SEQUENCE, n_features))
            # x = input_layer

            model = tf.keras.Sequential()

            model.add(tf.keras.layers.InputLayer(
                (settings.INPUT_SEQUENCE, n_features)))
            model.add(tf.keras.layers.LSTM(100, return_sequences=True))
            model.add(tf.keras.layers.LSTM(100, return_sequences=True))
            model.add(tf.keras.layers.LSTM(50))
            model.add(tf.keras.layers.Dense(8, activation='relu'))
            model.add(tf.keras.layers.Dense(20, activation='linear'))

            # for i in range(settings.N_LAYERS):
            #     if i == 0:
            #         x = tf.keras.layers.LSTM(
            #             settings.INPUT_NEURONS, return_sequences=True)(x)
            #         if settings.BIDIRECTIONAL:
            #             x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            #                 settings.INPUT_NEURONS, return_sequences=True))(x)
            #     else:
            #         x = tf.keras.layers.LSTM(
            #             settings.INPUT_NEURONS, return_sequences=True)(x)
            #         if settings.BIDIRECTIONAL:
            #             x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            #                 settings.INPUT_NEURONS, return_sequences=True))(x)

            #     # Add dropout after each layer
            #     x = tf.keras.layers.Dropout(settings.DROPOUT)(x)

            # # Move the following lines outside of the loop

            # output_layer = tf.keras.layers.Dense(2)(x)

            # Create the model
            # model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

            # Compile the model with the specified settings
            model.compile(loss=settings.LOSS, metrics=settings.METRICS,
                          optimizer=settings.OPTIMIZER)

            model.summary()

            cls.network = model

            cls._save_model()

    @classmethod
    async def train(cls, symbol: str, progress_callback=None):
        try:
            if cls.network is None:
                raise RuntimeError(
                    'A critical error occured training this model')

            log_path = os.path.join(
                settings.NUERON_PATH, settings.MODEL_PATH, settings.BASE_LOG_FOLDER)

            history_path = os.path.join(
                settings.NUERON_PATH, settings.MODEL_PATH, settings.BASE_HISTORY_FOLDER)

            check_point_path = os.path.join(
                settings.NUERON_PATH, settings.MODEL_PATH, settings.BASE_CHECKPOINT_FOLDER)

            # define a custom Tensorboard callback for logging during training
            tensorboard_callback = CustomTensorBoard(
                log_dir=log_path, update_freq="batch")

            check_pointer = tf.keras.callbacks.ModelCheckpoint(os.path.join(
                check_point_path, settings.MODEL_NAME), save_weights_only=True, save_best_only=True, verbose=1)

            # Create early stopping callback
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, verbose=1, mode='min')

            num_days = 30  # Change this to the desired number of days

            # Create an empty DataFrame
            dataframe = pd.DataFrame(columns=["prices", "times"])

            # Get the current date
            current_date = datetime.now()

            # Iterate for the specified number of days
            for day in range(num_days):
                # Calculate start and end dates in epoch seconds
                start_date = current_date - timedelta(days=day)
                if day == 0:  # First iteration
                    end_epoch = "latest"
                else:
                    end_date = start_date + timedelta(days=1)
                    end_epoch = int(end_date.timestamp())
                start_epoch = int(start_date.timestamp())

                symbol_data = await DerivService.get_instance().get_history_for_symbol(symbol=symbol, start=start_epoch, end=end_epoch, count=5000)

                # Create a DataFrame for the current day's data
                day_dataframe = pd.DataFrame({
                    "prices": symbol_data["history"]["prices"],
                    "times": symbol_data["history"]["times"]
                })

                # Append the current day's data to the existing DataFrame
                dataframe = dataframe.append(day_dataframe, ignore_index=True)

                # Sort the DataFrame by "times" column and keep only the "prices" column
                dataframe = dataframe.sort_values(by="times", ascending=True)[
                    settings.FEATURE_COLUMNS]

                if day < num_days - 1:
                    # Sleep if needed to avoid overloading the API
                    time.sleep(1)  # You can adjust the sleep duration

            print("First Element:\n", dataframe.head(1), "\nLast Element:\n", dataframe.tail(
                1), "\nLength of the DataFrame:", len(dataframe), "Number of rows in the DataFrame:", dataframe.shape[0])

            # preprocess the data and create a new model
            X_train, X_test, y_train, y_test = cls._preprocess_data(
                train_data=dataframe)

            print(
                f"X-Train: {X_train[0]}, X-Test: {X_test[0]}, y-Train: {y_train[0]}, y-Test: {y_test[0]}")

            # train the model and log results using the Tensorboard callback
            history = cls.network.fit(
                verbose=1,
                x=X_train,
                y=y_train,
                epochs=settings.EPOCHS,
                batch_size=settings.BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=[tensorboard_callback, check_pointer, early_stop, progress_callback])

            # convert the history.history dict to a pandas DataFrame:
            hist_df = pd.DataFrame(history.history)

            # save to json:
            hist_json_file = os.path.join(history_path, settings.HISTORY_FILE)
            with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f)

            # save the trained model for future use
            cls._save_model()

        except Exception as e:
            # raise an error if there was an issue with training
            raise RuntimeError(
                "An error occurred training the model " + str(e))

    @classmethod
    def _save_model(cls):
        path = os.path.join(settings.NUERON_PATH,
                            settings.MODEL_PATH, settings.MODEL_NAME)
        # Save the trained machine learning model to a file
        cls.network.save(path)
        logger.info(f'Model saved at ${datetime.now()}')

    @classmethod
    def _preprocess_data(cls, train_data: pd.DataFrame):
        # Convert the DataFrame to a NumPy array
        np_data = train_data.to_numpy()

        # Use MinMaxScaler to scale the data
        scaler = MinMaxScaler()
        np_scaled = scaler.fit_transform(np_data)

        # Determine the training data length based on the split ratio
        train_data_length = int(len(np_scaled) * (1 - settings.TEST_SIZE))

        # Split the data into training and testing sets
        train_data = np_scaled[:train_data_length]
        test_data = np_scaled[train_data_length:]

        # Partition the data into input (x) and output (y) sequences
        x_train, y_train = partition_dataset(train_data)
        x_test, y_test = partition_dataset(test_data)

        return x_train, x_test, y_train, y_test
