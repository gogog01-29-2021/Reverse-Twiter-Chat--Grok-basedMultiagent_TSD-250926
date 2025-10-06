import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

import csp
from csp import PushMode

from config import TransportAddressBuilder
from core.types.exchange import Exchange
from dsm.core.pubsub_base import Subscriber, PushInputAdapterManager, ZmqSubTransport, SubTransport
from dsm.publisher.execution_report_publisher import ExecutionReportPublisher
from dsm.utils.conversion_utils import epoch_ms_to_datetime

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ExecutionReportSubscriber(Subscriber):
    """
    Subscriber that listens to external order update topics and supports both:
    - CSP adapters (adapter.push_tick)
    - Plain callback (handler(msg))
    """

    def __init__(self, sub_transport: SubTransport, handler: Callable[[dict], None] = None):
        super().__init__(sub_transport)
        self._handler = handler  # optional external callback

    def _handle_message(self, topic: str, msg: dict, **kwargs):
        try:
            msg["time_exchange"] = epoch_ms_to_datetime(msg["time_exchange"])
            msg["time_received"] = epoch_ms_to_datetime(msg["time_received"])
            msg["time_published"] = epoch_ms_to_datetime(msg["time_published"])
        except Exception as e:
            logging.warning("ExternalOrderUpdateSubscriber: Failed to parse timestamps for topic %s: %s", topic, e)

        adapter = kwargs.get("adapter", None)
        if adapter:
            adapter.push_tick(msg)
        elif self._handler:
            self._handler(msg)
        else:
            logging.warning("ExternalOrderUpdateSubscriber: No adapter or handler set for topic: %s", topic)


class ExecutionReportAdapterManager(PushInputAdapterManager):
    """
    Manager that wires ZMQ subscriber to CSP input adapters for external order updates.
    """

    def __init__(self, zmq_connect_addr: str):
        super().__init__(ExecutionReportSubscriber(ZmqSubTransport(zmq_connect_addr=zmq_connect_addr)))

    def subscribe(self, topic: str, out_type=dict, push_mode: PushMode = PushMode.NON_COLLAPSING):
        # out_type=ExecutionReport
        return super().subscribe(topic, out_type=out_type, push_mode=push_mode)


@csp.graph
def external_order_graph():
    logging.info("Building CSP graph for external order updates")
    topic = ExecutionReportPublisher.topic(Exchange.COINBASE)
    manager = ExecutionReportAdapterManager(
        zmq_connect_addr=TransportAddressBuilder.external_order_update(Exchange.COINBASE, "tcp://localhost"))
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
