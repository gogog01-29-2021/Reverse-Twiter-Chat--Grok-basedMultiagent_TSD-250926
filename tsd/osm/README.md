Order Service Manager

# config.py

import os
from dotenv import load_dotenv

# 환경변수 로드

load_dotenv()

# 기본 환경설정

TIMEOUT = float(os.getenv("OSM_HTTP_TIMEOUT", "1.0"))
ENV = os.getenv("OSM_ENV", "local")
IS_TEST_MODE = os.getenv("TEST", "false").lower() == "true"

# 거래소 API 키 설정

EXCHANGE_CREDENTIALS = {
"BINANCE": {
"API_KEY": (
"api-key"
if IS_TEST_MODE else os.getenv("BINANCE_API_KEY")
),
"SECRET": (
"secret"
if IS_TEST_MODE else os.getenv("BINANCE_SECRET")
),
},
"COINBASE": {
"API_KEY": (
"api-key"
if IS_TEST_MODE else os.getenv("COINBASE_API_KEY")
),
"SECRET": (
"secret"
if IS_TEST_MODE else os.getenv("COINBASE_SECRET")
),
"PASSPHRASE": ("pass-phrase" if IS_TEST_MODE else os.getenv("COINBASE_PASSPHRASE")),
},
}

# REST API 엔드포인트 설정 (TEST 여부에 따라 자동 분기)

EXCHANGE_ENDPOINTS = {
"BINANCE": "https://testnet.binance.vision/api/v3/order" if IS_TEST_MODE else "https://api.binance.com/api/v3/order",
"BYBIT": "https://api-testnet.bybit.com/v5/order/create" if IS_TEST_MODE else "https://api.bybit.com/v5/order/create",
"COINBASE": "https://api-public.sandbox.exchange.coinbase.com/orders" if IS_TEST_MODE else "https://api.exchange.coinbase.com/orders", # OKX는 필요시 추가
}

# .env

BINANCE_API_KEY=api-key
BINANCE_SECRET=secret

BYBIT_API_KEY=api-key
BYBIT_SECRET=secret

COINBASE_API_KEY=api-key
COINBASE_SECRET=secrte
COINBASE_PASSPHRASE=pass-phrase

OSM_HTTP_TIMEOUT=1
ENV=local

BINANCE_WS_URL=wss://stream.binance.com:9443/ws

TEST=true
