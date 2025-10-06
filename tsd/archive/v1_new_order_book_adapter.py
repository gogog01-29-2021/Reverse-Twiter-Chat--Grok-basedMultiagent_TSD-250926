import asyncio
import json
import threading
import time

import csp
import websockets
import logging
from datetime import datetime, timedelta
from csp import ts
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def
from core.types.instrument import Spot, Currency
from core.types.marketdata import MarketOrder, OrderBook
from core.types.trade import OrderSide

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("CoinbaseAdapter")

# Optional: create a file handler for detailed debug logs if needed
debug_handler = logging.FileHandler('debug.log')
debug_handler.setLevel(logging.DEBUG)  # Change to INFO to reduce logs
debug_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
debug_handler.setFormatter(debug_formatter)
log.addHandler(debug_handler)


class OrderBookState:
    def __init__(self, instr):
        self.instr = instr
        self.bids = {}
        self.asks = {}

    def apply_snapshot(self, bids, asks):
        log.debug(f"[{self.instr}] Applying snapshot with {len(bids)} bids and {len(asks)} asks")
        self.bids = {float(price): float(size) for price, size in bids}
        self.asks = {float(price): float(size) for price, size in asks}

    def apply_changes(self, changes):
        log.debug(f"[{self.instr}] Applying {len(changes)} changes")
        for side, price_str, size_str in changes:
            price = float(price_str)
            size = float(size_str)
            book = self.bids if side == 'bid' else self.asks
            if size == 0.0:
                book.pop(price, None)
            else:
                book[price] = size

    def get_sorted_bids(self):
        return [
            MarketOrder(
                instr=self.instr,
                exchange="coinbase",
                order_side=OrderSide.BUY,
                price=p,
                qty=s,
                time_exchange=int(time.time_ns()),
                time_received=int(time.time_ns())
            )
            for p, s in sorted(self.bids.items(), reverse=True)
        ]

    def get_sorted_asks(self):
        return [
            MarketOrder(
                instr=self.instr,
                exchange="coinbase",
                order_side=OrderSide.SELL,
                price=p,
                qty=s,
                time_exchange=int(time.time_ns()),
                time_received=int(time.time_ns())
            )
            for p, s in sorted(self.asks.items())
        ]


class CoinbasePushAdapterImpl(PushInputAdapter):
    def __init__(self, product_id):
        super().__init__()
        self._product_id = product_id
        base_str, term_str = product_id.split("-")
        self._instr = Spot(base=Currency[base_str], term=Currency[term_str])
        self._state = OrderBookState(self._instr)
        self._ws_thread = None
        self._running = False
        self._loop = asyncio.new_event_loop()

    def start(self, start_time, end_time):
        self._running = True
        self._ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._ws_thread.start()
        log.info(f"[{self._product_id}] Adapter started")

    def stop(self):
        self._running = False
        if self._ws_thread:
            self._ws_thread.join()

    def _run_ws_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_websocket())

    async def _run_websocket(self):
        uri = 'wss://advanced-trade-ws.coinbase.com'
        subscribe_message = {
            "type": "subscribe",
            "product_ids": [self._product_id],
            "channel": "level2"
        }

        log.info(f"[{self._product_id}] Connecting to WebSocket...")
        while self._running:
            try:
                async with websockets.connect(uri, ping_interval=30, ping_timeout=20) as ws:
                    log.info(f"[{self._product_id}] WebSocket connected.")
                    await ws.send(json.dumps(subscribe_message))
                    log.info(f"[{self._product_id}] Subscription sent.")

                    while self._running:
                        complete_msg = await ws.recv()

                        # Log the complete raw WebSocket message
                        log.debug(f"[{self._product_id}] Received raw WebSocket message: {complete_msg}")

                        # Parse and log the parsed message content
                        try:
                            msg_obj = json.loads(complete_msg)
                            log.debug(f"[{self._product_id}] Parsed message: {json.dumps(msg_obj, indent=2)}")
                        except json.JSONDecodeError as e:
                            log.error(f"[{self._product_id}] Failed to decode message: {e}")

                        # Check if the subscription was acknowledged
                        if 'type' in msg_obj and msg_obj['type'] == 'subscriptions':
                            log.info(f"[{self._product_id}] Subscription confirmed: {msg_obj}")
                        self._process_message(msg_obj)

            except Exception as e:
                log.error(f"[{self._product_id}] WebSocket error: {e}, reconnecting...")
                await self.reconnect(uri)

    async def reconnect(self, uri):
        backoff_time = 2
        while self._running:
            try:
                log.info(f"[{self._product_id}] Reconnecting...")
                await asyncio.sleep(backoff_time)
                async with websockets.connect(uri, ping_interval=30, ping_timeout=20) as ws:
                    log.info(f"[{self._product_id}] WebSocket reconnected.")
                    return
            except Exception as e:
                log.error(f"[{self._product_id}] Reconnection failed: {e}")
                backoff_time *= 2  # Exponential backoff

    def _process_message(self, msg_obj):
        msg_type = msg_obj.get("type")

        if msg_type == "snapshot":
            updates = msg_obj.get("updates", [])
            if updates:
                log.info(f"[{self._product_id}] Snapshot received with {len(updates)} updates.")
                bids = [(u["price_level"], u["new_quantity"]) for u in updates if u["side"] == "bid"]
                asks = [(u["price_level"], u["new_quantity"]) for u in updates if u["side"] == "ask"]
                log.debug(f"[{self._product_id}] Snapshot Bids: {bids[:5]} Asks: {asks[:5]}")
                self._state.apply_snapshot(bids, asks)
                self._emit_orderbook()
            else:
                log.warning(f"[{self._product_id}] Snapshot received but no updates.")

        elif msg_type == "update":
            changes = [(u["side"], u["price_level"], u["new_quantity"]) for u in msg_obj.get("updates", [])]
            if changes:
                log.debug(f"[{self._product_id}] Processing updates: {changes[:5]}...")
                self._state.apply_changes(changes)
                self._emit_orderbook()
            else:
                log.warning(f"[{self._product_id}] No updates in the message.")

    def _emit_orderbook(self):
        # Print the current state to check if it's being updated
        log.debug(f"[{self._product_id}] Current bids: {self._state.bids}")
        log.debug(f"[{self._product_id}] Current asks: {self._state.asks}")

        # Create OrderBook and emit
        book = OrderBook(
            instr=self._instr,
            bids=self._state.get_sorted_bids(),
            asks=self._state.get_sorted_asks(),
            time_exchange=int(time.time_ns()),
            time_received=int(time.time_ns())
        )
        log.info(f"[{self._product_id}] Emitting OrderBook: {len(book.bids)} bids / {len(book.asks)} asks")
        self.push_tick(book)


# Definition of push adapter
CoinbaseOrderBookAdapter = py_push_adapter_def(
    name="CoinbaseOrderBookAdapter",
    adapterimpl=CoinbasePushAdapterImpl,
    out_type=ts[OrderBook],
    product_id=str
)


@csp.graph
def my_graph():
    btc = CoinbaseOrderBookAdapter("BTC-USD", push_mode=csp.PushMode.NON_COLLAPSING)
    eth = CoinbaseOrderBookAdapter("ETH-USD", push_mode=csp.PushMode.NON_COLLAPSING)
    csp.print("BTC OrderBook", btc)
    csp.print("ETH OrderBook", eth)


if __name__ == "__main__":
    log.info("Starting CSP runtime...")
    csp.run(
        my_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(minutes=5),
        realtime=True
    )
