from deriv_api import DerivAPI
from config.settings import DERIV_APP_ID, DERIV_ENDPOINT


class DerivService:
    _instance = None

    def __init__(self):
        self.api: DerivAPI = None  # Initialize the API instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(cls):
        # Note: Instead of initializing cls.api, you should initialize self.api
        cls._instance.api = DerivAPI(app_id=DERIV_APP_ID)

    async def get_history_for_symbol(self, symbol, start, end, count):
        response = await self.api.ticks_history({"ticks_history": symbol, "style": "ticks", "end": end, "start": start, "count": count})

        return response

    async def get_ticks_for_symbol(self, symbol):
        ticks = await self.api.subscribe({"ticks": symbol, "subscribe": 1})

        return ticks
