import sys
from datetime import datetime, timedelta

import csp
from csp.adapters.websocket import RawTextMessageMapper, Status, WebsocketAdapterManager


@csp.node
def send_subscription_request_on_connect(status_stream: csp.ts[Status], subscription_message: str) -> csp.ts[str]:
    """
    Sends a subscription request message to the websocket server upon successful connection.
    """
    if csp.ticked(status_stream) and status_stream.status_code == 0:
        return subscription_message


@csp.graph
def market_data_websocket_client(uri: str):
    """
    Connects to a market data websocket server, subscribes to data, and prints incoming messages and connection status.
    """
    print(f"Attempting to connect to market data websocket at {uri}")
    ws_manager = WebsocketAdapterManager(uri)

    # Subscribe to incoming raw text market data messages
    market_data_messages = ws_manager.subscribe(str, RawTextMessageMapper())

    # Monitor websocket connection status
    connection_status = ws_manager.status()

    # Send subscription request once connected
    subscription_msg = "SUBSCRIBE:market_data_feed"
    ws_manager.send(send_subscription_request_on_connect(connection_status, subscription_msg))

    csp.print("WebSocket Status", connection_status)
    csp.print("Market Data Message", market_data_messages)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: market_data_websocket_client.py <websocket_uri>")
        sys.exit(1)

    csp.run(
        market_data_websocket_client,
        starttime=datetime.utcnow(),
        endtime=timedelta(minutes=1),
        realtime=True,
        uri=sys.argv[1],
    )
