from core.types.exchange import Exchange
from osm.managers.binance import BinanceOrderManager
from osm.managers.bybit import BybitOrderManager
from osm.managers.coinbase import CoinbaseOrderManager

class OrderServiceManager:
    def __init__(self, config: dict):
        self.config = config

    def get(self, exchange: Exchange):
        if exchange == Exchange.BINANCE:
            return BinanceOrderManager(*self.config["binance"])
        elif exchange == Exchange.BYBIT:
            return BybitOrderManager(*self.config["bybit"])
        elif exchange == Exchange.COINBASE:
            return CoinbaseOrderManager(*self.config["coinbase"])
        # elif exchange == Exchange.OKX:
        #     return OkxOrderManager(*self.config["OKX"])
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")