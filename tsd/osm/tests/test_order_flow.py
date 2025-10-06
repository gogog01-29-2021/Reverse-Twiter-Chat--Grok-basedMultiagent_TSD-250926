import time
import pytest

from core.types.instrument import Currency, Spot
from core.types.exchange import Exchange
from core.types.trade import ExternalOrder, OrderSide, OrderType, TimeInForce
from osm.managers.binance import BinanceOrderManager
from osm.managers.coinbase import CoinbaseOrderManager
from config import TEST_EXCHANGE_CREDENTIALS
from osm.utils.clientIdGen import ClientOrderIdGenerator


@pytest.mark.asyncio
async def test_binance_rest_order():
    creds = TEST_EXCHANGE_CREDENTIALS["BINANCE"]
    manager = BinanceOrderManager(creds["API_KEY"], creds["SECRET"])

    order = ExternalOrder(
        instr=Spot(base=Currency.ETH, term=Currency.BTC),
        exchange=Exchange.BINANCE,
        order_side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=0.03,
        qty=0.1,
        time_in_force=TimeInForce.GTC,
        time_sent=time.time_ns(),
        client_order_id= ClientOrderIdGenerator.generate(Exchange.BINANCE)
    )
    print("order:", order)

    result = await manager.send_order(order)
    print("✅ SEND ORDER RESULT:", result)

    assert result["exchange"] == "BINANCE"
    assert result["symbol"] == "ETHBTC"
    assert result["side"] == "BUY"
    assert result["type"] == "LIMIT"
    assert result["status"] in ["NEW", "PARTIALLY_FILLED", "FILLED"]
    assert float(result["price"]) == pytest.approx(order.price, rel=1e-8)
    assert float(result["origQty"]) == pytest.approx(order.qty, rel=1e-8)


@pytest.mark.asyncio
async def test_coinbase_rest_order():
    creds = TEST_EXCHANGE_CREDENTIALS["COINBASE"]
    manager = CoinbaseOrderManager(
        creds["API_KEY"], creds["SECRET"], creds["PASSPHRASE"]
    )

    order = ExternalOrder(
        instr=Spot(base=Currency.ETH, term=Currency.BTC),
        exchange=Exchange.COINBASE,
        order_side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=99999.0,
        qty=0.001,
        time_in_force=TimeInForce.IOC,
        time_sent=time.time_ns(),
        client_order_id= ClientOrderIdGenerator.generate(Exchange.COINBASE)
    )
    print("order:", order)

    result = await manager.send_order(order)
    print("✅ SEND ORDER RESULT:", result)

    assert result["exchange"] == "COINBASE"
    assert result["product_id"] == "ETH-BTC"
    assert result["side"] == "buy"
    assert result["type"] == "limit"
    assert result["time_in_force"] == "IOC"
    assert float(result["price"]) == pytest.approx(order.price, rel=1e-8)
    assert float(result["size"]) == pytest.approx(order.qty, rel=1e-8)
    assert result["status"] in ["pending", "open", "done"]