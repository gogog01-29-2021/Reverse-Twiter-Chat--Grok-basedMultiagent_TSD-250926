## Table of Contents

- [Table of Contents](#table-of-contents)
- [`csp.feedback`](#cspfeedback)
- [`csp.DelayedEdge`](#cspdelayededge)

CSP has two methods of late binding a time series to an edge.
- `csp.feedback`: Be used to create connections from downstream nodes to upstream nodes in a graph without introducing a cycle.
- `csp.DelayedEdge`: Be used when a single edge may come from many possible sources and will be bound after its passed as an argument to some other node.

## `csp.feedback`

CSP graphs are always *directed acyclic graphs* (DAGs); however, there are some occasions where you may want to feed back the output of a downstream node into a node at a prior [rank](../stage2/CSP%20Graph.md#Graph-Propagation-and-Single-Dispatch).

This is usually the case when a node is making a decision that depends on the result of its previous action.
For example, consider a `csp.graph` that simulates systemic risk propagation in an interbank network. We have a system of banks interconnected through various exposures—bank A, bank B, ..., and each has a default probability and capital buffer.
We have a node `adjust_capital_ratio` which simulates capital adequacy adjustments for a particular bank, say bank A, based on current exposure levels and perceived risk from counterparties. Then we have a node `simulate_contagion_impact` which computes the expected impact of bank A’s state on other banks in the network—this includes factors like mark-to-market losses or funding stress due to interbank exposures.
The output of `simulate_contagion_impact`, representing the updated risk levels across the network, will feed back into the `adjust_capital_ratio` node in the next iteration. This means the systemic risk model becomes capable of capturing endogenous feedback loops, such as how one institution’s stress triggers changes across the network that in turn influence the original institution’s stability.
This kind of modeling allows us to simulate dynamic feedback mechanisms in financial systems, where a small local shock can propagate and amplify through interconnected decisions and reactions.

For use cases like this, the `csp.feedback` construct exists. It allows us to pass the output of a downstream node to one upstream so that a recomputation is triggered on the next [engine cycle](../stage2/CSP%20Graph.md#Graph-Propagation-and-Single-Dispatch).
Using `csp.feedback`, one can wire a feedback as an input to a node, and effectively bind the actual edge that feeds it later in the graph.

> [!IMPORTANT]


> A graph containing one or more `csp.feedback` edges is still acyclic. The feedback connection will trigger a recomputation of the upstream node on the next engine cycle, which will be at the same engine time as the current cycle. Internally `csp.feedback` creates a pair of input and output adapters that are bound together.

- **`csp.feedback(ts_type)`**: `ts_type` is the type of the timeseries (ie int, str).
  This returns an instance of a feedback object, which will *later* be bound to a downstream edge.
  - **`out()`**: this method returns the timeseries edge which can be passed as an input to your node
  - **`bind(ts)`**: this method is called to bind a downstream edge as the source of the feedback

Let us demonstrate the usage of `csp.feedback` using our systemic risk propagation example above. The graph code would look something like this:

```python
import csp
from csp import ts

@csp.node
def adjust_capital_ratio(network_risk: ts[float], idiosyncratic_factors: ts[float]) -> ts[float]:
    """
    Simulates a bank's adjustment to its capital ratio.
    - network_risk: systemic stress from the financial network
    - idiosyncratic_factors: internal or firm-specific risk buffers
    """
    # Capital ratio decreases with systemic risk, increases with internal stability
    capital = 0.1 - network_risk + idiosyncratic_factors
    return capital

@csp.node
def simulate_contagion_impact(capital_ratio: ts[float], exposure_factor: ts[float]) -> ts[float]:
    """
    Estimates how much risk this bank contributes back to the system.
    - capital_ratio: the bank’s solvency measure
    - exposure_factor: how interconnected the bank is with the system
    """
    # If the capital ratio is low, the bank poses more contagion risk
    contagion_risk = exposure_factor * max(0.0, 0.1 - capital_ratio)
    return contagion_risk

@csp.graph
def systemic_risk_dynamics():
    # Feedback loop for systemic risk
    systemic_risk_fb = csp.feedback(float)

    # Bank adjusts capital based on current systemic risk
    capital_ratio = adjust_capital_ratio(systemic_risk_fb.out(), csp.const(0.03))

    # Bank's condition impacts the rest of the system
    contagion = simulate_contagion_impact(capital_ratio, csp.const(0.5))

    # Close the feedback loop
    systemic_risk_fb.bind(contagion)
```

We can visualize the graph using `csp.show_graph`. We see that it remains acyclic, but since the `FeedbackOutputDef` is bound to the `FeedbackInputDef` any output tick will loop back in at the next engine cycle.

![Output generated by show_graph](feedback-graph.png)

## `csp.DelayedEdge`

The delayed edge is similar to `csp.feedback` in the sense that it's a time series which is bound after its declared. Delayed edges must be bound *exactly* once and will raise an error during graph building if unbound.
Delayed edges can also not be used to create a cycle; if the edge is being bound to a downstream output, `csp.feedback` must be used instead. Any cycle will be detected by the CSP engine and raise a runtime error.

Delayed edges are useful when the exact input source needed is not known until graph-time; for example, you may want to subscribe to a list of data feeds which will only be known when you construct the graph.
They are also used by some advanced `csp.baselib` utilities like `DelayedCollect` and `DelayedDemultiplex` which help with input and output data processing.

An example usage of `csp.DelayedEdge` is below:

```python
import csp
from csp import ts
from typing import Tuple

@csp.node
def compute_momentum(price: ts[float], window: float = 5.0) -> ts[float]:
    """
    Compute momentum as the price change over a time window.
    """
    return price.diff(window)

@csp.node
def select_high_momentum_symbol(aapl_mom: ts[float], tsla_mom: ts[float]) -> ts[str]:
    """
    Select the stock with stronger positive momentum.
    If momentum is equal or both are negative, default to AAPL.
    """
    def selector(aapl: float, tsla: float) -> str:
        if aapl > tsla:
            return "AAPL"
        elif tsla > aapl:
            return "TSLA"
        else:
            return "AAPL"
    
    return csp.map2(aapl_mom, tsla_mom, selector)

@csp.node
def process_price(price: ts[float]):
    """
    Output the selected price stream.
    """
    csp.print("Selected price", price)

@csp.graph
def market_data_router():
    # Simulate price streams (replace with csp.sub for live data)
    aapl_price = csp.sub("AAPL.price")
    tsla_price = csp.sub("TSLA.price")

    # Compute momentum for each stock
    aapl_momentum = compute_momentum(aapl_price)
    tsla_momentum = compute_momentum(tsla_price)

    # Decide which stock has higher momentum
    selected_symbol = select_high_momentum_symbol(aapl_momentum, tsla_momentum)

    # Create filter conditions
    is_aapl = selected_symbol.map(lambda s: s == "AAPL")
    is_tsla = selected_symbol.map(lambda s: s == "TSLA")

    # DelayedEdge used to bind price stream dynamically
    delayed = csp.DelayedEdge(csp.ts[float])

    # Bind to the appropriate price stream based on symbol selection
    delayed.bind(
        csp.select(is_aapl, aapl_price)
            .else_select(is_tsla, tsla_price)
    )

    # Process selected price
    process_price(delayed)
```

Executing this graph will give:

```
2020-01-01 00:00:00 Selected price: 150.25
2020-01-01 00:00:01 Selected price: 150.40
2020-01-01 00:00:02 Selected price: 150.35
...
```
