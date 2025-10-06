#
# The csp.DynamicBasket construct is used to dynamically manage a set of live timeseries
#
from datetime import datetime, timedelta

import csp
import csp.showgraph
from csp import ts


@csp.node
def price_feed(symbol: str, base_price: float) -> ts[float]:
    """
    Simulate price feed for a given stock.
    """
    yield csp.now(), base_price
    yield csp.now() + timedelta(seconds=1), base_price + 1.5
    yield csp.now() + timedelta(seconds=2), base_price + 3.0


@csp.node
def dynamic_portfolio_selection(price: ts[float], symbol: ts[str]) -> csp.DynamicBasket[str, float]:
    """
    Dynamically construct a portfolio based on live price signals.
    """
    if csp.ticked(price) and csp.valid(symbol):
        if price > 101.0:
            # Add the symbol to the dynamic portfolio
            csp.output({symbol: price})
        elif price < 99.0:
            # Remove the symbol from the dynamic portfolio
            csp.remove_dynamic_key(symbol)


@csp.node
def monitor_dynamic_portfolio(portfolio: csp.DynamicBasket[str, float]):
    """
    Monitor changes in the dynamic portfolio: log additions and removals.
    """
    if csp.ticked(portfolio.shape):
        for symbol in portfolio.shape.added:
            print(f'{csp.now()} - Symbol {symbol} added to the portfolio')
        for symbol in portfolio.shape.removed:
            print(f'{csp.now()} - Symbol {symbol} removed from the portfolio')

    if csp.ticked(portfolio):
        for symbol, price in portfolio.tickeditems():
            print(f'{csp.now()} - {symbol} latest price: {price}')


@csp.graph
def dynamic_portfolio_graph():
    # Simulate two stock price feeds
    aapl_price = price_feed("AAPL", 100.0)
    tsla_price = price_feed("TSLA", 100.0)

    # Create a dynamic portfolio for each stock
    aapl_portfolio = dynamic_portfolio_selection(aapl_price, csp.const("AAPL"))
    tsla_portfolio = dynamic_portfolio_selection(tsla_price, csp.const("TSLA"))

    # Merge the dynamic baskets into a single portfolio
    combined_portfolio = csp.merge_dynamic_baskets(aapl_portfolio, tsla_portfolio)

    # Monitor portfolio changes
    monitor_dynamic_portfolio(combined_portfolio)


def main():
    show_graph = False
    if show_graph:
        csp.showgraph.show_graph(dynamic_portfolio_graph)
    else:
        csp.run(dynamic_portfolio_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=False)


if __name__ == "__main__":
    main()
