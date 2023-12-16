
import os
import time
import pytz
import logging
import requests
import pandas as pd
from datetime import datetime
import yfinance as yahooFinance
from routes.client import HttpClient
from modules.logger import AppLogger
from apps.neural_network import settings
from config.settings import YAHOO_FINANCE_URL

logger = AppLogger('bot_logger', logging.DEBUG)


class ExchangeMarket:
    def __init__(self):
        pass

    def get_symbols(self):
        response = requests.get(YAHOO_FINANCE_URL)
        dfs = pd.read_html(response.content)

        currencies_df = dfs[0]
        currencies = list(currencies_df['Symbol'].str.upper())

        logger.info(
            f"Retreived all available forex currencies - count: {len(currencies)}")

        return currencies

    def get_symbol_data(self, symbol: str, interval: str, period: str, output=5000) -> pd.DataFrame:

        if settings.USE_DEBUG_API:

            forex_data: pd.DataFrame = yahooFinance.download(
                symbol, period=period, interval=interval,)

            forex_data = forex_data.reset_index()

            forex_data = forex_data.dropna()

            logger.info(
                f"Retreived all available market data for the symbol {symbol} - rows: {forex_data.shape[0]} columns: {forex_data.shape[1]}")

            return forex_data
        else:

            http_client = HttpClient(
                f"https://api.twelvedata.com/time_series?outputsize={output}&symbol={symbol}&interval={interval}&apikey={settings.TIME_SERIES_API_KEY}&timezone=UTC")

            response = http_client.get()
            json = response.json()

            forex_data = pd.DataFrame(json['values'])

            forex_data = forex_data.dropna()

            forex_data.rename(columns={
                              'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

            print(forex_data)

            return forex_data

    def is_market_open(self):
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        forex_open = False

        if now.weekday() < 4:  # Monday to Thursday
            forex_open = True
        elif now.weekday() == 4:  # Friday
            time_close = datetime.combine(now.date(), time(
                hour=16, minute=0, second=0, microsecond=0), tz)
            if now <= time_close:
                forex_open = True
        elif now.weekday() == 5 or now.weekday() == 6:  # Saturday or Sunday
            forex_open = False

        return True
