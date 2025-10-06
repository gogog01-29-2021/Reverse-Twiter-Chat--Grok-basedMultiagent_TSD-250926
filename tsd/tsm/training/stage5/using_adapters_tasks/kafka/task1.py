import os
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.adapters.kafka import (
    DateTimeType,
    JSONTextMessageMapper,
    KafkaAdapterManager,
    ProtoMessageMapper,
    RawTextMessageMapper,
)


# Structure to represent JSON market data message
class MarketDataMessage(csp.Struct):
    symbol: str
    price: float
    size: int


# Structure to represent ProtoBuf FX snapshot message
class FXSnapshotMessage(csp.Struct):
    timestamp: csp.Struct  # Nested timestamp structure
    pair: str
    lp: str
    tier: float
    price: float


class FXSnapshotMultiMessage(csp.Struct):
    t_entry: csp.Struct
    pair: str
    lp: str
    tier: float
    price: float


@csp.graph
def market_data_consumer_graph():
    """Consumes market data messages from Kafka and prints them."""
    broker = "localhost:9092"
    kafka = KafkaAdapterManager(broker)
    topic = "marketdata-events"

    field_map = {"symbol": "symbol", "price": "price", "size": "size"}

    msg_mapper = JSONTextMessageMapper()

    data = kafka.subscribe(ts_type=MarketDataMessage, msg_mapper=msg_mapper, topic=topic, field_map=field_map)
    csp.print("Market Data", data)


class MyData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime


class EnrichedData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    b2: bool
    i2: int
    d2: float
    s2: str
    dt2: datetime
    extra_metric: float
    extra_tag: str


@csp.node
def current_time(x: ts[object]) -> ts[datetime]:
    if csp.ticked(x):
        return csp.now()


@csp.graph
def market_data_producer_graph():
    """Simulates a market data producer that publishes JSON messages to Kafka."""
    broker = "localhost:9092"
    kafka = KafkaAdapterManager(broker)
    topic = "marketdata-events"

    b = csp.merge(
        csp.timer(timedelta(seconds=0.2), True),
        csp.delay(csp.timer(timedelta(seconds=0.2), False), timedelta(seconds=0.1)),
    )
    i = csp.count(csp.timer(timedelta(seconds=0.15)))
    d = csp.count(csp.timer(timedelta(seconds=0.2))) / 2.0
    s = csp.sample(csp.timer(timedelta(seconds=0.4)), csp.const("TRADE"))
    dt = current_time(b)
    struct = MyData.collectts(b=b, i=i, d=d, s=s, dt=dt)

    msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

    field_map = {"b": "b2", "i": "i2", "d": "d2", "s": "s2", "dt": "dt2"}

    kafka.publish(msg_mapper=msg_mapper, topic=topic, x=struct, field_map=field_map, key="events1")

    enriched_data = EnrichedData.collectts(
        b=b, i=i, d=d, s=s, dt=dt, b2=struct.b, i2=struct.i, d2=struct.d, s2=struct.s, dt2=struct.dt
    )

    csp.print("Enriched Market Data", enriched_data)

    subscribed_data = kafka.subscribe(ts_type=EnrichedData, msg_mapper=msg_mapper, topic=topic, key="events1")
    csp.print("Subscribed Market Data", subscribed_data)

    status = kafka.status()
    csp.print("Kafka Status", status)


@csp.graph
def fx_snapshot_graph():
    """Consumes FX snapshot ProtoBuf messages from Kafka and prints them."""
    broker = "localhost:9092"
    kafka = KafkaAdapterManager(broker)

    topic = "fx-snapshot-topic"

    field_map = {
        "t_entry": {"timestamp": {"seconds": "seconds", "nanos": "nanos"}},
        "pair": "pair",
        "lp": "lp",
        "tier": "tier",
        "price": "price",
    }

    msg_mapper = ProtoMessageMapper(
        proto_directory="/tmp", proto_filename="fxspotstream.proto", proto_message="Snapshot"
    )

    data = kafka.subscribe(ts_type=FXSnapshotMessage, msg_mapper=msg_mapper, topic=topic, key="events")
    csp.print("FX Snapshot", data)


@csp.graph
def fx_snapshot_multi_subscribers_graph():
    """Consumes FX snapshot messages with multiple subscribers."""
    broker = "localhost:9092"
    kafka = KafkaAdapterManager(broker)

    topic = "fx-snapshot-topic"

    msg_mapper = ProtoMessageMapper(
        proto_directory="/tmp", proto_filename="fxspotstream.proto", proto_message="Snapshot"
    )

    data1 = kafka.subscribe(ts_type=FXSnapshotMultiMessage, msg_mapper=msg_mapper, topic=topic)
    data2 = kafka.subscribe(ts_type=FXSnapshotMultiMessage, msg_mapper=msg_mapper, topic=topic)

    csp.print("FX Snapshot - Subscriber 1", data1)
    csp.print("FX Snapshot - Subscriber 2", data2)


class RawTextMessage(csp.Struct):
    text: str


@csp.graph
def kerberos_market_data_graph():
    """Consumes market data with Kerberos-authenticated Kafka."""
    broker = "securebroker:9093"
    kafka = KafkaAdapterManager(
        broker,
        auth=True,
        sasl_kerberos_keytab=os.getenv("KEYTAB_LOCATION"),
        sasl_kerberos_principal=os.getenv("USER_PRINCIPAL_NAME"),
        ssl_ca_location=os.getenv("KAFKA_PEM_FILE"),
    )
    topic = "secured-marketdata"

    msg_mapper = JSONTextMessageMapper()
    subscribed_data = kafka.subscribe(
        ts_type=RawTextMessage, msg_mapper=msg_mapper, topic=topic, key="events1", reset_offset="none"
    )
    csp.print("Kerberos Secured Market Data", subscribed_data)

    status = kafka.status()
    csp.print("Kafka Status", status)


@csp.graph
def raw_text_consumer_graph():
    """Consumes raw text messages from Kafka and prints them."""
    broker = "securebroker:9093"
    kafka = KafkaAdapterManager(
        broker,
        auth=True,
        sasl_kerberos_keytab=os.getenv("KEYTAB_LOCATION"),
        sasl_kerberos_principal=os.getenv("USER_PRINCIPAL_NAME"),
        ssl_ca_location=os.getenv("KAFKA_PEM_FILE"),
    )
    topic = "barra_factor_return_topic"

    field_map = {"": "text"}

    msg_mapper = RawTextMessageMapper()
    subscribed_data = kafka.subscribe(
        ts_type=RawTextMessage,
        msg_mapper=msg_mapper,
        topic=topic,
        key="barra",
        field_map=field_map,
        push_mode=csp.PushMode.NON_COLLAPSING,
        reset_offset="earliest",
    )
    csp.print("Barra Factor Return Data", subscribed_data)

    status = kafka.status()
    csp.print("Kafka Status", status)


if __name__ == "__main__":
    # Choose which graph to run
    csp.run(market_data_consumer_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    # csp.run(market_data_producer_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    # csp.run(fx_snapshot_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    # csp.run(fx_snapshot_multi_subscribers_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    # csp.run(kerberos_market_data_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    # csp.run(raw_text_consumer_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
