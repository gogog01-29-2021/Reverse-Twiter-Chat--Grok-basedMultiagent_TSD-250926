from datetime import timedelta

import csp
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def


from csp.impl.adaptermanager import AdapterManagerImpl
from datetime import datetime
import threading, asyncio, json, time, websockets

from config import EXCHANGE_CONFIG
from core.types.instrument import Instrument
from core.types.marketdata import MarketOrder, OrderBook

import logging

from core.utils.auth import subscription_token

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger("CoinbaseAdapter")


class OrderBookState:
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = {}  # price: size
        self.asks = {}

    def apply_snapshot(self, bids, asks):
        self.bids = {float(price): float(size) for price, size in bids}
        self.asks = {float(price): float(size) for price, size in asks}

    def apply_changes(self, changes):
        for side, price_str, size_str in changes:
            price = float(price_str)
            size = float(size_str)
            book = self.bids if side == 'buy' else self.asks
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size

    def get_sorted_bids(self):
        return [MarketOrder(price=p, size=s) for p, s in sorted(self.bids.items(), reverse=True)]

    def get_sorted_asks(self):
        return [MarketOrder(price=p, size=s) for p, s in sorted(self.asks.items())]


class CoinbaseAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, product_ids, api_key, private_key):
        super().__init__(engine)
        self._product_ids = product_ids
        self._api_key = api_key
        self._secret_key = private_key
        self._inputs = {}
        self._orderbooks = {pid: OrderBookState(pid) for pid in product_ids}
        self._running = False
        self._thread = None

    def register_input_adapter(self, product_id, adapter):
        self._inputs.setdefault(product_id, []).append(adapter)

    def start(self, starttime, endtime):
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        self._started_event.set()  # signal feed is ready to push

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def process_next_sim_timeslice(self, now):
        return None

    def _run_async_loop(self):
        asyncio.run(self._run_websocket())

    async def _run_websocket(self):
        URI = 'wss://advanced-trade-ws.coinbase.com'
        jwt_token = subscription_token(self._api_key, self._secret_key)

        subscribe_message = {
            "type": "subscribe",
            "channel": "level2",
            "product_ids": self._product_ids,
            "jwt": jwt_token
        }

        while self._running:
            try:
                log.info("Connecting to Coinbase Advanced Trade WebSocket...")
                async with websockets.connect(URI, ping_interval=None) as ws:
                    log.info("WebSocket connection established.")
                    await ws.send(json.dumps(subscribe_message))
                    log.info("Subscription message sent.")

                    while self._running:
                        msg = await ws.recv()
                        msg_obj = json.loads(msg)
                        log.debug(f"Received message: {msg_obj}")

                        if msg_obj.get("channel") != "l2_data":
                            log.debug("Ignoring non-order book message.")
                            continue

                        for event in msg_obj.get("events", []):
                            product_id = event.get("product_id")
                            updates = event.get("updates", [])
                            event_type = event.get("type")

                            if product_id not in self._orderbooks:
                                continue

                            if event_type == "snapshot":
                                self._handle_snapshot(product_id, updates)
                                log.info(f"Got snapshot for {product_id} with {len(updates)} levels")
                                log.debug(f"First few: {updates[:5]}")
                            elif event_type == "update":
                                self._handle_update(product_id, updates)

            except Exception as e:
                log.error(f"[WebSocket Error] {e}, reconnecting...")
                await asyncio.sleep(2)

    def _handle_snapshot(self, product_id, updates):
        bids, asks = [], []
        for update in updates:
            price = update["price_level"]
            size = update["new_quantity"]
            side = update["side"]
            if side == "bid":
                bids.append([price, size])
            else:
                asks.append([price, size])
        self._orderbooks[product_id].apply_snapshot(bids, asks)
        self._emit_orderbook(product_id)

    def _handle_update(self, product_id, updates):
        changes = []
        for update in updates:
            price = update["price_level"]
            size = update["new_quantity"]
            side = update["side"]
            changes.append([side, price, size])
        self._orderbooks[product_id].apply_changes(changes)
        self._emit_orderbook(product_id)

    def _emit_orderbook(self, product_id):
        state = self._orderbooks[product_id]
        book = OrderBook(
            instr=Instrument(symbol=product_id),
            bids=state.get_sorted_bids(),
            asks=state.get_sorted_asks(),
            time_exchange=int(time.time_ns()),  # you can later parse from event_time
            time_received=int(time.time_ns())
        )

        log.info(f"Emitting OrderBook for {product_id}: "
                 f"{len(state.bids)} bids / {len(state.asks)} asks")

        for adapter in self._inputs.get(product_id, []):
            adapter.push_tick(book)
            log.info(f"Pushed tick to CSP for {product_id}")

    def _parse_time(self, msg):
        try:
            time_str = msg.get("time")
            if time_str:
                return int(datetime.fromisoformat(time_str.replace("Z", "+00:00")).timestamp() * 1e9)
        except Exception:
            pass
        return int(time.time_ns())


class CoinbaseAdapterManager:
    def __init__(self, product_ids, api_key: str, private_key_path: str):
        self._product_ids = product_ids
        self._api_key = api_key
        self._private_key_path = private_key_path

    def subscribe(self, product_id, push_mode):
        return CoinbasePushAdapter(self, product_id, push_mode=push_mode)

    def _create(self, engine, memo):
        return CoinbaseAdapterManagerImpl(engine, self._product_ids, self._api_key, self._private_key_path)



class CoinbasePushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, product_id):
        super().__init__()
        manager_impl.register_input_adapter(product_id, self)


CoinbasePushAdapter = py_push_adapter_def(
    name='CoinbasePushAdapter',
    adapterimpl=CoinbasePushAdapterImpl,
    out_type=ts[OrderBook],
    manager_type=CoinbaseAdapterManager,
    product_id=str
)


@csp.graph
def my_graph():
    symbols = ["ETH-USD", "BTC-USD"]
    api_key = EXCHANGE_CONFIG["coinbase"]["api_key"]
    private_key_path = EXCHANGE_CONFIG["coinbase"]["secret_key"]

    manager = CoinbaseAdapterManager(symbols, api_key, private_key_path)

    for symbol in symbols:
        ob = manager.subscribe(symbol, push_mode=csp.PushMode.LAST_VALUE)
        csp.print(f"{symbol} OrderBook", ob)


if __name__ == "__main__":
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=60),
        realtime=True
    )
