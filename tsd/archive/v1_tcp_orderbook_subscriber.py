#!/usr/bin/env python3
import zmq
import logging
import time
import threading

from config import EXCHANGE_CONFIG

# Configure logging to DEBUG for detailed output.
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

class Subscriber:
    def __init__(self, exchange, zmq_port):
        """
        Initialize the subscriber:
          - Connects to the specified port.
          - Subscribes to all messages.
          - Sets a reception timeout to periodically check the running flag.
        """
        self.exchange = exchange
        self.zmq_port = zmq_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.zmq_port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Set a reception timeout (in milliseconds) so the loop can exit if no message arrives.
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.running = False
        self.thread = None
        logging.info("Subscriber initialized and connected to tcp://localhost:%s", self.zmq_port)

    def subscribe_loop(self):
        """
        Main loop for receiving and processing messages.
        Runs in a dedicated thread.
        """
        logging.info("Subscription loop started.")
        while self.running:
            try:
                message = self.socket.recv_json()
                logging.debug("Raw message received: %s", message)
                topic = message.get("topic", "Unknown Topic")
                data = message.get("data", {})

                # Process only messages with topics starting with our prefix.
                print(topic)
                if not topic.startswith(f"ORDERBOOK_{self.exchange}"):
                    logging.debug("Ignoring message with topic: %s", topic)
                    continue

                if "bidPrices" in data and "askPrices" in data:
                    best_bid = data["bidPrices"][0] if data.get("bidPrices") and len(data["bidPrices"]) > 0 else "N/A"
                    best_ask = data["askPrices"][0] if data.get("askPrices") and len(data["askPrices"]) > 0 else "N/A"
                    # Extract the three timestamps from the published data.
                    timeExchange = data.get("timeExchange", "N/A")
                    timeReceived = data.get("timeReceived", "N/A")
                    timePublished = data.get("timePublished", "N/A")
                    logging.info(
                        "Order book received: Topic: %s, Exchange: %s, Symbol: %s, Best Bid: %s, Best Ask: %s, timeExchange: %s, timeReceived: %s, timePublished: %s",
                        topic,
                        data.get("exchange", "Unknown"),
                        data.get("symbol", "Unknown"),
                        best_bid,
                        best_ask,
                        timeExchange,
                        timeReceived,
                        timePublished
                    )
                else:
                    logging.warning("Received message without complete order book data: %s", message)
            except zmq.error.Again:
                logging.debug("No message received in the last timeout period.")
                continue
            except Exception as e:
                logging.error("Error processing message: %s", e)
                time.sleep(0.5)
        logging.info("Subscription loop exiting.")

    def start(self):
        """
        Start the subscription in a separate thread and log the start.
        """
        self.running = True
        self.thread = threading.Thread(target=self.subscribe_loop)
        self.thread.daemon = True
        self.thread.start()
        logging.info("Subscription started.")

    def end(self):
        """
        Stop the subscription:
          - Set the running flag to False.
          - Join the thread.
          - Close the socket and terminate the context.
          - Log that the subscription has ended.
        """
        logging.info("Stopping subscription...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.socket.close()
        self.context.term()
        logging.info("Subscription ended.")


if __name__ == "__main__":
    exchange = "bybit"
    subscriber = Subscriber(
        exchange=EXCHANGE_CONFIG[exchange]["exchange"],
        zmq_port=EXCHANGE_CONFIG[exchange]["orderbook_port"]
    )

    subscriber.start()
    # Let the subscriber run for 5 seconds (adjust as needed), then end.
    time.sleep(5)
    subscriber.end()
