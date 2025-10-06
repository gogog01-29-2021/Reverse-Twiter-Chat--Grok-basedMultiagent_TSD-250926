import datetime
import hashlib
import logging
import os
import threading
import time
from collections import defaultdict

import csp
import jwt
import websocket
from csp import ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

import config
from archive.v1_messaging import Streamer, dumps, loads
from core.types.position import Position


class PositionAdapterManager:
    def __init__(self):
        pass

    def subscribe(self, symbol, push_mode=csp.PushMode.NON_COLLAPSING):
        return PositionPushAdapter(self, symbol, push_mode=push_mode)

    def _create(self, engine, memo):
        impl = PositionAdapterManagerImpl(engine)
        PositionAdapterManager.impl_instance = impl  # Global holder
        return impl


class PositionAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine):
        super().__init__(engine)
        self._inputs = {}

    def register_input_adapter(self, symbol, adapter):
        if symbol not in self._inputs:
            self._inputs[symbol] = []
        self._inputs[symbol].append(adapter)

    def push_position(self, data: dict):
        for adapter in self._inputs.get(data["product_id"], []):
            adapter.push_tick(data)

    def start(self, starttime, endtime):
        # Required by CSP runtime (no-op is fine for real-time feed)
        print("PositionAdapterManagerImpl started.")

    def stop(self):
        print("PositionAdapterManagerImpl stopped.")

    def process_next_sim_timeslice(self, now):
        return None


class PositionPushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, symbol):
        manager_impl.register_input_adapter(symbol, self)
        super().__init__()


PositionPushAdapter = py_push_adapter_def(
    "PositionPushAdapter", PositionPushAdapterImpl, ts[dict], PositionAdapterManager, symbol=str)


class CoinbasePositionStreamer(Streamer):

    def __init__(self, ws_url, api_key, secret_key, symbols, exchange, csp_adapter=None, save_mode=True):
        super().__init__(ws_url)
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols
        self.exchange = exchange
        self.csp_adapter = csp_adapter
        self.save_mode = save_mode

        self.positions = defaultdict(float)
        self.seen_orders = set()  # Track processed order IDs

    def subscribe(self, ws):
        """
        Authenticate and subscribe to level2 channels for given symbols.
        """
        timestamp = int(time.time())
        payload = {
            "iss": "coinbase-cloud",
            "nbf": timestamp,
            "exp": timestamp + 120,
            "sub": self.api_key,
        }
        headers = {
            "kid": self.api_key,
            "nonce": hashlib.sha256(os.urandom(16)).hexdigest()
        }
        token = jwt.encode(payload, self.secret_key, algorithm="ES256", headers=headers)
        message = {
            "type": "subscribe",
            "channel": "user",
            "jwt": token
        }
        ws.send(dumps(message))
        logging.info("%s: Sent subscription message on channel %s", self.__class__.__name__, "user")

    def on_message(self, ws, message):
        logging.debug("%s: WebSocket message received: %s", self.__class__.__name__, message)
        if not isinstance(message, str):
            logging.debug("%s: Received non-string message: %s", self.exchange, message)
            return
        try:
            timeReceived = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            data = loads(message)
            self.update_position(data, timeReceived)
        except Exception as e:
            logging.error("%s: Error processing WebSocket message: %s", self.exchange, e)

    def logging_loop(self):
        pass

    def update_position(self, data, timeReceived):
        if "events" in data:
            for event in data["events"]:
                if event.get("type", "").lower() == "update":
                    for order in event.get("orders", []):
                        if order.get("status") == "FILLED":
                            order_id = order["order_id"]
                            if order_id in self.seen_orders:
                                continue  # Skip duplicate processing

                            self.seen_orders.add(order_id)  # Mark as seen

                            symbol = order["product_id"]
                            side = order["order_side"]  # 'BUY' or 'SELL'
                            qty = float(order["cumulative_quantity"])

                            # Compute delta position
                            if side == "BUY":
                                self.positions[symbol] += qty
                            elif side == "SELL":
                                self.positions[symbol] -= qty

                            net_qty = self.positions[symbol]

                            # Push net position to CSP adapter
                            if self.csp_adapter:
                                self.csp_adapter.push_position(order)

    def start_streaming(self):
        self.ws_app = websocket.WebSocketApp(
            self.ws_url,
            on_open=lambda ws: self.subscribe(ws),
            on_message=self.on_message
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        logging.info("%s: Started WebSocket streaming for symbols: %s", self.__class__.__name__, self.symbols)
        return self.ws_thread

    def start(self, block=True):
        self.start_streaming()
        logging.info("%s publisher is running.", self.__class__.__name__)
        if block:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.end()

    def end(self):
        logging.info("Stopping %s publisher...", self.__class__.__name__)
        if self.ws_app is not None:
            self.ws_app.keep_running = False
            self.ws_app.close()
        if self.ws_thread is not None:
            self.ws_thread.join(timeout=2)

        logging.info("%s publisher stopped.", self.__class__.__name__)


@csp.graph
def live_position_graph():
    manager = PositionAdapterManager()
    symbols = ["BTC-USD", "ETH-USD"]
    for symbol in symbols:
        data = manager.subscribe(symbol, push_mode=csp.PushMode.LAST_VALUE)
        csp.print(f"Position {symbol}", data)


def main():
    RUN_TIME = 180  # in seconds

    def run_engine():
        csp.run(live_position_graph, starttime=datetime.datetime.utcnow(), endtime=datetime.timedelta(seconds=RUN_TIME),
                realtime=True)

    engine_thread = threading.Thread(target=run_engine)
    engine_thread.start()

    # Wait for the adapter manager impl to initialize
    while not hasattr(PositionAdapterManager, 'impl_instance'):
        time.sleep(0.5)

    publisher = CoinbasePositionStreamer(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        api_key=config.EXCHANGE_CONFIG["coinbase"]["api_key"],
        secret_key=config.EXCHANGE_CONFIG["coinbase"]["secret_key"],
        symbols=["BTC-USD", "ETH-USD"],
        exchange="COINBASE",
        csp_adapter=PositionAdapterManager.impl_instance
    )

    publisher.start()
    time.sleep(RUN_TIME)
    publisher.end()
    engine_thread.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
