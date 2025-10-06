import uuid
import random
import string

from core.types.exchange import Exchange

class ClientOrderIdGenerator:
    PREFIXES = {
        Exchange.BINANCE: "bin",
        Exchange.COINBASE: "cb",
        Exchange.BYBIT: "byb",
        Exchange.OKX: "okx",
    }

    @classmethod
    def generate(cls, exchange: Exchange) -> str:
        prefix = cls.PREFIXES.get(exchange, "gen")  # fallback prefix
        
        if exchange == Exchange.COINBASE:
            # Coinbase는 UUID 필수
            return str(uuid.uuid4())
        else:
            # 다른 거래소는 prefix-랜덤조합
            random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
            return f"{prefix}-{random_part}"