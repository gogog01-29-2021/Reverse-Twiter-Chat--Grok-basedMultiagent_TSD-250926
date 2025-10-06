# Configuration Guide for `tsd` Project

This guide explains how to configure the project using `config.py` and a `.env` file. Proper configuration is essential for running the application smoothly in different environments.

---

## File Structure

```
./tsd/
├── config.py
├── .env          # Create this file based on the template provided in config.py
└── (other files...)
```

---

## Step 1: Setting Up `.env` File

1. Create a `.env` file in the root directory (`./tsd`).
2. Open `config.py` and scroll to the bottom where the environment variable **template** is provided.
3. Copy the template and fill in the appropriate values.

### Example `.env` Template

```
TEST=true
# === Environment Variables ===

# === General Settings ===
OSM_HTTP_TIMEOUT=1.0
OSM_ENV=local

# === Binance Configuration ===
BINANCE_PORT=5551
# BINANCE_URL=https://api.binance.com/api/v3/order
BINANCE_URL=https://testnet.binance.vision/api/v3/order
BINANCE_WS_URL=wss://stream.binance.com:9443/ws
BINANCE_API_KEY=gaodof9ZIbrXnRgBpfw09Qv7fHFqUgOjV9W0jh2idYWqTA5JK5VS8qfYH5gdlRCL
BINANCE_SECRET=XKqSJe0XeLcpxRrAdKHSHT9N5YQvVvYuvWdf6VIfChcVglGTjbq3x0jeC7MhmGrK

# === Coinbase Configuration ===
COINBASE_PORT=5552
# COINBASE_URL=https://api.coinbase.com/api/v3/brokerage/orders
COINBASE_URL=https://api-public.sandbox.exchange.coinbase.com/orders
COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com
COINBASE_API_KEY=6d9a5112095690dd7c6c070167d45eaa
COINBASE_SECRET_KEY=1mRJICbEx9b8SMBgTitlmUmmCavzkHdGAJGPqvXlI4z8gT4B3EMiscFX96fzO6RI2QXLRM2BrU9rZ+COYUOkGA==
COINBASE_PASSPHRASE=fbaquantTSD2

# === Bybit Configuration ===
BYBIT_PORT=5553
BYBIT_URL=https://api.bybit.com/v5/order/create
BYBIT_WS_URL=wss://stream.bybit.com/realtime
BYBIT_API_KEY=y8j3Q7ml6zrK1gOOyD
BYBIT_SECRET_KEY=gTWvsG0RIDCQPbNW8uDjUv0JAJ520dpv22MW

# === OKX Configuration ===
OKX_PORT=5554
OKX_URL=https://www.okx.com/api/v5/trade/order
OKX_WS_URL=wss://ws.okx.com:8443/ws/v5/public
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret
```

---

## Step 2: How `config.py` Loads `.env`

- `config.py` automatically reads values from the `.env` file using a library like `dotenv` or similar.
- These values are then used to configure database connections, external APIs, and other environment-dependent settings.

---

## Final Checklist

- [ ] `.env` file is created in `./tsd`.
- [ ] All required variables are filled out based on the `config.py` template.
- [ ] `.env` is added to `.gitignore` to avoid committing sensitive information.
- [ ] Restart the application after updating `.env`.

---

For any issues regarding configuration, please contact the project maintainer or check `config.py` for additional comments and required fields.
