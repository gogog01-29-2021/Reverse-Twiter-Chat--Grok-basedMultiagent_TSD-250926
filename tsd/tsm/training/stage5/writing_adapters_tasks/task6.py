"""
This is a simple example demonstrating an output adapter that collects simulated price data into an output buffer.
"""

from datetime import datetime, timedelta
from json import dumps

import csp
from csp import ts
from csp.impl.outputadapter import OutputAdapter
from csp.impl.wiring import py_output_adapter_def


class PriceData(csp.Struct):
    symbol: str
    price: float


class PriceBufferWriterAdapterImpl(OutputAdapter):
    def __init__(self, output_buffer):
        super().__init__()
        self.input_buffer = []
        self.output_buffer = output_buffer

    def start(self):
        # Clear the output buffer at the start of the graph
        self.output_buffer.clear()

    def stop(self):
        # Serialize the collected price data into JSON and store in output_buffer
        data = dumps([d.__dict__ for d in self.input_buffer])
        self.output_buffer.append(data)

    def on_tick(self, time, value):
        self.input_buffer.append(value)


PriceBufferWriterAdapter = py_output_adapter_def(
    name="PriceBufferWriterAdapter",
    adapterimpl=PriceBufferWriterAdapterImpl,
    input=ts[PriceData],
    output_buffer=list,
)


output_buffer = []


@csp.graph
def price_graph():
    symbols = ["AAPL", "IBM", "TSLA"]
    initial_prices = [100.0, 150.0, 200.0]

    # Simulate price data (static list for this example)
    price_data = [
        PriceData(symbol=symbols[i % len(symbols)], price=initial_prices[i % len(symbols)] + i * 0.5)
        for i in range(5)
    ]

    curve = csp.curve(
        data=[(timedelta(seconds=i + 1), d) for i, d in enumerate(price_data)],
        typ=PriceData,
    )

    csp.print("Writing price data to buffer", curve)

    PriceBufferWriterAdapter(curve, output_buffer=output_buffer)


def main():
    csp.run(
        price_graph,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=6),
        realtime=True,
    )
    print("Output buffer (JSON serialized price data):")
    print(output_buffer)


if __name__ == "__main__":
    main()
