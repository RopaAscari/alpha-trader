import json
from django.db import models


class Prediction():
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, symbol, predictions, highest_prediction_price, lowest_prediction_price, prediction_interval):
        self.symbol = symbol
        self.prediction_interval = prediction_interval
        self.lowest_prediction_price = lowest_prediction_price
        self.highest_prediction_price = highest_prediction_price
        self.predictions = predictions

    def __repr__(self) -> str:
        return f"{type(self).__name__}(symbol={self.symbol}, predictions={self.predictions}, prediction_interval={self.prediction_interval})"

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    def __str__(self):
        return self.card_number
