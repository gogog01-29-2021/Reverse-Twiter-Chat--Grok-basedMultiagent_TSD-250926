#
# The csp.DelayedEdge construct is used to bind to a time series dynamically
#
from datetime import datetime, timedelta

import csp
import csp.showgraph
from csp import ts


class Order(csp.Struct):
    order_id: int
    symbol: str
    price: float
    qty: int
    side: str


@csp.node
def price_feed_aapl() -> ts[float]:
    """
    Simulate price feed for AAPL.
    """
    yield csp.now(), 150.0
    yield csp.now() + timedelta(seconds=1), 151.0
    yield csp.now() + timedelta(seconds=2), 152.5


@csp.node
def price_feed_tsla() -> ts[float]:
    """
    Simulate price feed for TSLA.
    """
    yield csp.now(), 700.0
    yield csp.now() + timedelta(seconds=1), 702.0
    yield csp.now() + timedelta(seconds=2), 703.5


@csp.node
def compute_momentum(price: ts[float]) -> ts[float]:
    """
    Simple momentum calculation: first difference.
    """
    return price.diff()


@csp.node
def select_symbol(aapl_mom: ts[float], tsla_mom: ts[float]) -> ts[str]:
    """
    Select the stock with the stronger positive momentum.
    """
    def selector(aapl: float, tsla: float) -> str:
        if aapl > tsla:
            return "AAPL"
        else:
            return "TSLA"

    return csp.map2(aapl_mom, tsla_mom, selector)


@csp.node
def order_generator(selected_price: ts[float], selected_symbol: ts[str]) -> ts[Order]:
    """
    Generate an order for the selected stock based on its latest price.
    """
    with csp.state():
        s_order_id = 1

    if csp.ticked(selected_price) and csp.valid(selected_symbol):
        order = Order(
            order_id=s_order_id,
            symbol=selected_symbol,
            price=selected_price,
            qty=100,
            side="BUY"
        )
        print(f"{csp.now()} Sending order id:{order.order_id} symbol:{order.symbol} price:{order.price}")
        s_order_id += 1
        return order


@csp.graph
def dynamic_order_routing():
    # Price feeds
    aapl_price = price_feed_aapl()
    tsla_price = price_feed_tsla()

    # Compute momentums
    aapl_momentum = compute_momentum(aapl_price)
    tsla_momentum = compute_momentum(tsla_price)

    # Select stock symbol with higher momentum
    selected_symbol = select_symbol(aapl_momentum, tsla_momentum)

    # Conditions for selecting price feed
    is_aapl = selected_symbol.map(lambda s: s == "AAPL")
    is_tsla = selected_symbol.map(lambda s: s == "TSLA")

    # Use DelayedEdge to dynamically bind the selected price feed
    delayed_price = csp.DelayedEdge(csp.ts[float])
    delayed_price.bind(
        csp.select(is_aapl, aapl_price)
            .else_select(is_tsla, tsla_price)
    )

    # Generate orders based on selected symbol and price
    order_generator(delayed_price, selected_symbol)


def main():
    show_graph = False
    if show_graph:
        csp.showgraph.show_graph(dynamic_order_routing)
    else:
        csp.run(dynamic_order_routing, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=False)


if __name__ == "__main__":
    main()
