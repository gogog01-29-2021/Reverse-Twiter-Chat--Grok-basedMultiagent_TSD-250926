import logging
from datetime import datetime, timezone, timedelta

import csp
from csp import PushMode

from config import TransportAddressBuilder
from config.routing_config import get_instruments_by_port
from core.types.exchange import Exchange
from core.types.marketdata import OrderBook, MarketOrder, Side
from dsm.core.pubsub_base import PushInputAdapterManager, Subscriber, SubTransport, ZmqSubTransport
from dsm.publisher.order_book_publisher import OrderBookPublisher
from dsm.utils.conversion_utils import symbol_to_instrument, epoch_ms_to_datetime


class OrderBookSubscriber(Subscriber):
    def _handle_message(self, topic: str, msg: dict, **kwargs):

        adapter = kwargs["adapter"]
        try:
            exchange = Exchange(msg["exchange"])
            instrument = symbol_to_instrument(msg["symbol"])
            time_exchange = epoch_ms_to_datetime(msg["time_exchange"])
            time_received = epoch_ms_to_datetime(msg["time_published"])

            bids = [
                MarketOrder(instr=instrument, exchange=exchange, side=Side.BID,
                            price=price, qty=qty,
                            time_exchange=time_exchange, time_received=time_received)
                for price, qty in msg["bids"]
            ]
            asks = [
                MarketOrder(instr=instrument, exchange=exchange, side=Side.ASK,
                            price=price, qty=qty,
                            time_exchange=time_exchange, time_received=time_received)
                for price, qty in msg["asks"]
            ]

            order_book = OrderBook(instr=instrument, bids=bids, asks=asks,
                                   time_exchange=time_exchange, time_received=time_received)
            adapter.push_tick(order_book)
        except Exception as e:
            logging.error("OrderBookSubscriber: Failed to handle topic %s - %s", topic, e)


class OrderBookInputAdapterManager(PushInputAdapterManager):
    def __init__(self, sub_transport: SubTransport):
        super().__init__(OrderBookSubscriber(sub_transport))

    def subscribe(self, topic, out_type=OrderBook, push_mode=PushMode.NON_COLLAPSING):
        return super().subscribe(topic, out_type, push_mode)


@csp.graph
def my_graph():
    logging.info("Building CSP graph")
    port = 5000
    exchange_to_connect = Exchange.COINBASE
    instruments_to_publish = get_instruments_by_port(exchange_to_connect, port)
    zmq_connect_addr = TransportAddressBuilder.order_book(exchange_to_connect, instruments_to_publish, "tcp://localhost")
    for instrument in instruments_to_publish:
        topic = OrderBookPublisher.topic(Exchange.COINBASE, instrument)
        book = OrderBookInputAdapterManager(
            ZmqSubTransport(zmq_connect_addr=zmq_connect_addr)).subscribe(topic=topic)
        csp.print(f"{instrument} order book:", book)


def run_my_graph():
    csp.run(
        my_graph,
        starttime=datetime.now(timezone.utc),
        endtime=timedelta(seconds=5),
        realtime=True
    )


if __name__ == "__main__":
    run_my_graph()
