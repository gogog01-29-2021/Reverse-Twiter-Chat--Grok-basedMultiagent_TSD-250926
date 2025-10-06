import os

from dotenv import load_dotenv

from core.types.exchange import Exchange

# 환경변수 로드
load_dotenv()


def require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"[CONFIG] Missing required environment variable: {var_name}")
    return value


# 기본 환경설정
TIMEOUT = float(os.getenv("OSM_HTTP_TIMEOUT", "1.0"))
ENV = os.getenv("OSM_ENV", "local")

EXCHANGE_CONFIG = {
    Exchange.BINANCE: {
        "api_key": require_env("BINANCE_API_KEY"),
        "secret_key": require_env("BINANCE_SECRET"),
    },
    Exchange.COINBASE: {
        "api_key": require_env("COINBASE_API_KEY"),
        "secret_key": require_env("COINBASE_SECRET_KEY"),
        "passphrase": os.getenv("COINBASE_PASSPHRASE", ""),
    },
    Exchange.BYBIT: {
        "api_key": require_env("BYBIT_API_KEY"),
        "secret_key": require_env("BYBIT_SECRET_KEY"),
    },
    Exchange.OKX: {
        "api_key": require_env("OKX_API_KEY"),
        "secret_key": require_env("OKX_SECRET_KEY"),
    },
}

# IS_TEST_MODE = os.getenv("TEST", "false").lower() == "true"

TEST_EXCHANGE_ENDPOINTS = {
    "BINANCE": "https://testnet.binance.vision/api/v3/order",
    "BYBIT": "https://api-testnet.bybit.com/v5/order/create",
    "COINBASE": "https://api-public.sandbox.exchange.coinbase.com/orders",
    # OKX는 필요시 추가
}

TEST_EXCHANGE_CREDENTIALS = {
    "BINANCE": {
        "API_KEY": (
            "gaodof9ZIbrXnRgBpfw09Qv7fHFqUgOjV9W0jh2idYWqTA5JK5VS8qfYH5gdlRCL"

        ),
        "SECRET": (
            "XKqSJe0XeLcpxRrAdKHSHT9N5YQvVvYuvWdf6VIfChcVglGTjbq3x0jeC7MhmGrK"

        ),
    },
    "COINBASE": {
        "API_KEY": (
            "07aad0d29b481f995b951c9b9e115b5a"

        ),
        "SECRET": (
            "6IQj5y0CHu3Jk0IXaM+l8p3YNiQm9kDiO+dL9VUvslkmK1EzW/vnYg4VUz2Gch4aZJeWsaLPJ7C02sCWAnSjGw=="

        ),
        "PASSPHRASE": ("yia1vflt8fge"),
    },
}

"""
example of .env file
# === Environment Variables ===


# === General Settings ===
OSM_HTTP_TIMEOUT=1.0
OSM_ENV=local

# === Binance Configuration ===
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret

# === Coinbase Configuration ===
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

# === Bybit Configuration ===
BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET_KEY=your_bybit_secret

# === OKX Configuration ===
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret
"""
