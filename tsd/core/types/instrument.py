from datetime import datetime
from enum import StrEnum

import csp

from core.types.exchange import Exchange


class Instrument(csp.Struct):
    symbol: str


class Currency(StrEnum):
    USD = "USD"
    USDT = "USDT"
    USDC = "USDC"
    BTC = "BTC"
    ETH = "ETH"
    DAI = "DAI"
    SOL = "SOL"
    FDUSD = "FDUSD"
    XRP = "XRP"
    ZKJ = "ZKJ"
    DOGE = "DOGE"
    PEPE = "PEPE"
    SUI = "SUI"
    TRUMP = "TRUMP"


## Spot

class Spot(Instrument):
    base: Currency
    term: Currency

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol = f"{self.base}{self.term}"

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol


## Future

# TODO: complete Enums and Structs for futures
# https://developers.binance.com/docs/derivatives/coin-margined-futures/common-definition
# https://developers.binance.com/docs/derivatives/usds-margined-futures/common-definition

class SymbolType(StrEnum):
    DELIVERY = "DELIVERY"
    PERPETUAL = "PERPETUAL"


class ContractType(StrEnum):
    PERPETUAL = "PERPETUAL"


class Future(Instrument):
    exchange: Exchange
    underlying: Instrument
    contract_size: float
    contract_price_multiplier: float
    min_price_increment: float
    is_inverted: bool
    is_primary: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f""


## Option

# TODO: complete Enums and Structs
# https://developers.binance.com/docs/derivatives/option/common-definition

class OptionStyle(StrEnum):
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


class OptionType(StrEnum):
    CALL = "CALL"
    PUT = "PUT"


class SettlementType(StrEnum):
    PHYSICAL = "PHYSICAL"
    CASH = "CASH"


class Option(Instrument):
    underlying: Instrument
    option_style: OptionStyle
    option_type: OptionType
    settlement_type: SettlementType
    settlement_currency: Currency
    strike: float
    expiration_date: datetime
    settlement_date: datetime

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f""


if __name__ == "__main__":
    # instr = Instrument(symbol="DummyInstrument")
    # print(instr)
    #
    # btc_usd = Spot(base=Currency.BTC, term=Currency.USD)
    # print(btc_usd.term)
    # print(btc_usd.symbol)
    # print(btc_usd)
    print("BTC" in Currency)
    print(Currency.BTC in Currency)
