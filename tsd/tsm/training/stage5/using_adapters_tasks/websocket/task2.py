import math
from datetime import datetime, timedelta

import csp
from csp import ts
from csp.adapters.websocket import WebsocketTableAdapter

"""
To view the output, open the provided sample HTML (see below) in a browser
and connect to the websocket server.
"""


class MarketSignal(csp.Struct):
    symbol_id: int
    indicator_value: float
    indicator_scaled: float
    signal_strength: float
    timestamp: datetime


@csp.node
def signal_function(indicator_scaled: ts[float]) -> ts[float]:
    if csp.ticked(indicator_scaled):
        return math.sin(indicator_scaled)


@csp.node
def current_time(trigger: ts[bool]) -> ts[datetime]:
    if csp.ticked(trigger):
        return csp.now()


@csp.graph
def market_signal_graph(ws_port: int, num_symbols: int):
    """
    Generates simulated market signals for multiple symbols and publishes them via a websocket table.
    """
    tick_timer = csp.timer(timedelta(seconds=0.25))
    indicator_counter = csp.count(tick_timer)

    all_signals = []
    for symbol_id in range(1, num_symbols + 1):
        delay = 10.0 * (symbol_id / float(num_symbols))
        delayed_indicator = csp.delay(indicator_counter, timedelta(seconds=delay))
        scaled_indicator = delayed_indicator / math.pi
        signal_strength = signal_function(scaled_indicator)

        signal_data = MarketSignal.fromts(
            symbol_id=csp.const(symbol_id),
            indicator_value=csp.cast_int_to_float(indicator_counter),
            indicator_scaled=scaled_indicator,
            signal_strength=signal_strength,
            timestamp=current_time(tick_timer),
        )
        all_signals.append(signal_data)

    combined_signals = csp.flatten(all_signals)
    ws_adapter = WebsocketTableAdapter(ws_port)

    table = ws_adapter.create_table("market_signal_table", index="symbol_id")
    table.publish(combined_signals)

    csp.print("MarketSignal Data", combined_signals)


def main():
    ws_port = 7677
    num_symbols = 10
    csp.run(
        market_signal_graph,
        ws_port,
        num_symbols,
        starttime=datetime.utcnow(),
        endtime=timedelta(seconds=360),
        realtime=True,
    )


"""
Sample HTML to view the market signal data.

Replace 'server' with your hostname or IP address:

<html>
<head><title>Market Signal Viewer</title></head>
<body>
  <script>
    async function main() {
      let response = await fetch("http://server:7677/tables");
      let data = await response.json();

      let table = data.tables[0];
      let ws = new WebSocket(table.sub);

      ws.onmessage = (event) => {
          let msg = JSON.parse(event.data);
          console.log('Received Market Signal:', msg.data);
      }
    }

    main();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
