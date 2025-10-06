from typing import Dict, Tuple, List

from core.constants.spot import BTCUSD, ETHUSD, XRPUSD, SOLUSD
from core.types.exchange import Exchange
from core.types.instrument import Instrument

ORDER_BOOK_PORTS: Dict[Tuple[Exchange, Instrument], int] = {
    (Exchange.COINBASE, BTCUSD): 5000,
    (Exchange.COINBASE, XRPUSD): 5000,
    (Exchange.COINBASE, ETHUSD): 5001,
    (Exchange.COINBASE, SOLUSD): 5001,
    (Exchange.BINANCE, BTCUSD): 5010,
    (Exchange.BINANCE, XRPUSD): 5010,
    (Exchange.BINANCE, ETHUSD): 5011,
    (Exchange.BINANCE, SOLUSD): 5011,
}

EXTERNAL_ORDER_UPDATE_PORTS: Dict[Exchange, int] = {
    Exchange.COINBASE: 6000,
    Exchange.BINANCE: 6001,
}

POSITION_PORTS: Dict[Exchange, int] = {
    Exchange.COINBASE: 7000,
    Exchange.BINANCE: 7001,
}

class AddressBuilder:
    @staticmethod
    def websocket(exchange: Exchange) -> str:
        if exchange == Exchange.COINBASE:
            return "wss://advanced-trade-ws.coinbase.com"
        raise ValueError(f"No WebSocket address defined for {exchange}")

    @staticmethod
    def rest_order(exchange: Exchange) -> str:
        if exchange == Exchange.COINBASE:
            return "https://api.coinbase.com/v2/orders"
        elif exchange == Exchange.BINANCE:
            return "https://api.binance.com/api/v3/order"
        else:
            raise ValueError(f"No REST order endpoint defined for {exchange}")


class TransportAddressBuilder:
    @staticmethod
    def order_book(exchange: Exchange, instruments: List[Instrument], ip: str) -> str:
        ports = {
            ORDER_BOOK_PORTS.get((exchange, instr))
            for instr in instruments
            if (exchange, instr) in ORDER_BOOK_PORTS
        }
        ports.discard(None)
        if len(ports) != 1:
            raise ValueError(f"Expected all instruments to map to one unique port for {exchange}, got: {ports}")
        return f"{ip}:{ports.pop()}"

    @staticmethod
    def external_order_update(exchange: Exchange, ip: str) -> str:
        try:
            port = EXTERNAL_ORDER_UPDATE_PORTS[exchange]
            return f"{ip}:{port}"
        except KeyError:
            raise ValueError(f"No port assigned for external order updates for {exchange}")

    @staticmethod
    def position(exchange: Exchange, ip: str) -> str:
        try:
            port = POSITION_PORTS[exchange]
            return f"{ip}:{port}"
        except KeyError:
            raise ValueError(f"No port assigned for position updates for {exchange}")


# look up instruments by (exchange, port)
def get_instruments_by_port(exchange: Exchange, port: int) -> List[Instrument]:
    return [
        instrument for (ex, instrument), p in ORDER_BOOK_PORTS.items()
        if ex == exchange and p == port
    ]
