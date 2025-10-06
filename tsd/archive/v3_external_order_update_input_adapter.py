import logging
from datetime import datetime, timedelta, timezone

import csp
from csp import ts, PushMode
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.wiring import py_push_adapter_def

from core.types.exchange import Exchange
from dsm.utils.conversion_utils import epoch_ms_to_datetime
from archive.v3_order_book_input_adapter import ZmqSubscriber

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ExternalOrderUpdateSubscriber(ZmqSubscriber):
    def __init__(self, engine, zmq_url: str, exchange: Exchange):
        super().__init__(engine, zmq_url)
        self._exchange = exchange

    def handle_message(self, topic: str, msg: dict):
        try:
            msg["time_exchange"] = epoch_ms_to_datetime(msg["time_exchange"])
            msg["time_received"] = epoch_ms_to_datetime(msg["time_received"])
            msg["time_published"] = epoch_ms_to_datetime(msg["time_published"])
        except Exception as e:
            logging.warning(f"Failed to parse timestamps: {e}")

        self._adapter.push_tick(msg)


class ExternalOrderUpdateAdapterManager:
    def __init__(self, zmq_url: str, exchange: Exchange):
        self._zmq_url = zmq_url
        self._exchange = exchange

    def subscribe(self, topic: str, push_mode=PushMode.NON_COLLAPSING):
        return ExternalOrderUpdatePushAdapter(self, topic, push_mode=push_mode)

    def _create(self, engine, memo):
        return ExternalOrderUpdateSubscriber(engine, zmq_url=self._zmq_url, exchange=self._exchange)


class ExternalOrderUpdatePushAdapterImpl(PushInputAdapter):
    def __init__(self, manager_impl, topic: str):
        super().__init__()
        manager_impl.register_input_adapter(self, topic)
        logging.info(f"{self.__class__.__name__}::init {topic}")


ExternalOrderUpdatePushAdapter = py_push_adapter_def(
    "ExternalOrderUpdatePushAdapter",
    ExternalOrderUpdatePushAdapterImpl,
    ts[dict],
    ExternalOrderUpdateAdapterManager,
    topic=str,
)


@csp.graph
def external_order_graph():
    logging.info("Building CSP graph for external order updates")
    manager = ExternalOrderUpdateAdapterManager(
        zmq_url="tcp://localhost:5556",
        exchange=Exchange.COINBASE,
    )
    topic = f"ExternalOrderUpdate_{Exchange.COINBASE.value}"
    updates = manager.subscribe(topic)
    csp.print("Order update received:", updates)


def run_graph():
    csp.run(
        external_order_graph,
        starttime=datetime.now(timezone.utc),
        endtime=timedelta(seconds=3600),
        realtime=True,
    )


if __name__ == "__main__":
    run_graph()
