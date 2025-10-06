import asyncio
import logging
from typing import List, Optional

from archive.v1_messaging import Subscriber, loads

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class OrderBookSubscriber(Subscriber):
    def __init__(self, subjects: Optional[List[str]], nats_url: str = "nats://localhost:4222"):
        """
        Initialize the subscriber.

        Args:
            nats_url (str): NATS server URL.
            subjects (List[str]): NATS subjects to subscribe to (wildcards supported).
        """
        super().__init__(subjects, nats_url)

    async def _handle_message(self, msg):
        """
        Internal message handler.

        Args:
            msg (nats.aio.msg.Msg): Incoming NATS message.
        """
        try:
            data = loads(msg.data)
            exchange = data.get("exchange", "UNKNOWN")
            symbol = data.get("symbol", "UNKNOWN")

            bid_prices = data.get("bidPrices", [])
            ask_prices = data.get("askPrices", [])

            best_bid = bid_prices[0] if bid_prices else "N/A"
            best_ask = ask_prices[0] if ask_prices else "N/A"

            logging.info(f"[{exchange}][{symbol}] Best Bid: {best_bid} | Best Ask: {best_ask}")

        except Exception as e:
            logging.error("Error handling message on subject %s: %s", msg.topic, e)


if __name__ == "__main__":
    async def main():
        subscriber = OrderBookSubscriber(subjects=[
            "ORDERBOOK_COINBASE_BTC-USD",

        ])
        try:
            await subscriber.run()
        except KeyboardInterrupt:
            logging.info("Interrupted. Shutting down...")
            await subscriber.shutdown()


    asyncio.run(main())
