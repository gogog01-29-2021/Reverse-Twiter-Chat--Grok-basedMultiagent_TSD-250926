import logging
from typing import Dict, Any, Tuple

import requests

from config import TransportAddressBuilder, get_exchange_credentials
from core.types.exchange import Exchange
from core.types.instrument import Spot, Currency
from core.types.position import Position
from config.portfolio_config import Portfolio, supported_exchanges
from dsm.core.pubsub_base import ZmqSubTransport, Subscriber, PubTransport, ZmqPubTransport
from dsm.subscriber.execution_report_subscriber import ExecutionReportSubscriber
from dsm.utils.conversion_utils import now_epoch_ms


class PositionPublisher:
    def __init__(self, transport: PubTransport, portfolio: Portfolio):
        self._transport = transport
        self._portfolio = portfolio
        self._positions: Dict[Tuple[str, str], float] = {}  # (portfolio_id, symbol) -> qty
        self._initialize_positions()

    def _initialize_positions(self):
        logging.info("Fetching initial positions for portfolio %s", self._portfolio.name)

        for exchange in supported_exchanges(portfolio):
            creds = get_exchange_credentials(exchange)
            api_key, secret_key = creds["api_key"], creds["secret_key"]
            portfolio_id = self._portfolio.name

            try:
                if exchange == Exchange.COINBASE:
                    logging.warning("Skipping Coinbase position fetch until auth signature is implemented")
                    continue

                elif exchange == Exchange.BINANCE:
                    headers = {"X-MBX-APIKEY": api_key}
                    response = requests.get("https://api.binance.com/api/v3/account", headers=headers)
                    data = response.json().get("balances", [])
                    for item in data:
                        asset = item.get("asset")
                        qty = float(item.get("free", 0)) + float(item.get("locked", 0))
                        if qty > 0:
                            symbol = asset + "-USDT"
                            self._positions[(portfolio_id, symbol)] = qty
                            self._publish_position(portfolio_id, symbol)

            except Exception as e:
                logging.warning("Failed to fetch initial positions for %s: %s", exchange, e)

    def on_order_update(self, update: Dict[str, Any]):
        portfolio_id = update.get("retail_portfolio_id")
        side = update.get("order_side")
        order_status = update.get("status", "").lower()
        symbol = update.get("product_id")
        filled_qty = float(update.get("cumulative_quantity", 0))

        if not portfolio_id or not symbol or not side or order_status != "filled":
            return

        key = (portfolio_id, symbol)
        current_qty = self._positions.get(key, 0.0)

        if side.upper() == "BUY":
            self._positions[key] = current_qty + filled_qty
        elif side.upper() == "SELL":
            self._positions[key] = current_qty - filled_qty

        self._publish_position(portfolio_id, symbol)

    def _publish_position(self, portfolio_id: str, symbol: str):
        qty = self._positions[(portfolio_id, symbol)]
        try:
            instr = Spot(base=Currency[symbol.split("-")[0]], term=Currency[symbol.split("-")[1]])
        except KeyError:
            logging.warning("Unrecognized symbol format for position publishing: %s", symbol)
            return
        pos = Position(instr=instr, qty=qty, time_received=now_epoch_ms())
        topic = f"Position_{portfolio_id}_{symbol.replace('-', '')}"
        self._transport.send(topic, pos.__dict__)
        logging.info("Published position: %s", pos)


class PositionOrderUpdateSubscriber:
    def __init__(self, subscriber: Subscriber, publisher: PositionPublisher):
        self._subscriber = subscriber
        self._publisher = publisher

    def start(self, exchange: Exchange):
        topic = f"ExternalOrderUpdate_{exchange.value}"
        self._subscriber.subscribe(topic=topic)
        self._subscriber.start()

    def stop(self):
        self._subscriber.stop()


if __name__ == "__main__":
    transport = ZmqPubTransport(TransportAddressBuilder.position(Exchange.COINBASE, "tcp://0.0.0.0"), debug=True)
    portfolio = Portfolio.DEFAULT
    publisher = PositionPublisher(transport, portfolio)

    for exchange in supported_exchanges(portfolio):
        sub_transport = ZmqSubTransport(
            zmq_connect_addr=TransportAddressBuilder.external_order_update(exchange, "tcp://localhost"))
        subscriber = ExecutionReportSubscriber(
            sub_transport=sub_transport,
            handler=publisher.on_order_update)
        pos_sub = PositionOrderUpdateSubscriber(subscriber, publisher)
        pos_sub.start(exchange)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logging.info("Shutting down...")
