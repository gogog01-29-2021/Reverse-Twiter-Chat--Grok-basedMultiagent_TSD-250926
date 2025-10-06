from enum import StrEnum


class Exchange(StrEnum):
    COINBASE = "COINBASE"
    BYBIT = "BYBIT"
    BINANCE = "BINANCE"
    OKX = "OKX"
    UPBIT = "UPBIT"
    BITGET = "BITGET"
    MEXC = "MEXC"
    CDC = "CDC"  # Crypto.com Exchange
    HTX = "HTX"
    GATEIO = "GATEIO"


if __name__ == "__main__":
    coinbase = Exchange.COINBASE
    print(coinbase)
