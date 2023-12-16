
from typing import TypedDict, Literal


Symbols = Literal["EURUSD=X", "JPY=X"]


class SupportedCurrency:
    def __init__(self, symbol, name, debug_ticker=None):
        self.name = name
        self.symbol = symbol
        self.debug_ticker = debug_ticker
