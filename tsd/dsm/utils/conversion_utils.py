from datetime import datetime, timezone

from core.types.exchange import Exchange
from core.types.instrument import Instrument, Spot, Currency


def instrument_to_exchange_symbol(exchange: Exchange, instrument: Instrument) -> str:
    """
    Maps an instrument to its exchange-specific symbol representation.

    This function converts a normalized instrument, represented by a `Instrument` object
    into the format required by a specific exchange:

    - Binance: 'BTCUSD'
    - Coinbase: 'BTC-USD'
    - OKX: 'BTC-USD'
    - Bybit: 'BTCUSD'

    Args:
        exchange (Exchange): Enum representing the target exchange.
        instrument (Instrument): Object representing the instrument.

    Returns:
        str: Symbol string formatted for the specified exchange.

    Raises:
        ValueError: If the exchange is not supported.
    """
    if isinstance(instrument, Spot):
        if exchange == Exchange.BINANCE:
            return f"{instrument.base}{instrument.term}"  # Binance format: BTCUSD
        elif exchange == Exchange.COINBASE:
            return f"{instrument.base}-{instrument.term}"  # Coinbase format: BTC-USD
        elif exchange == Exchange.OKX:
            return f"{instrument.base}-{instrument.term}"  # OKX format: BTC-USD
        elif exchange == Exchange.BYBIT:
            return f"{instrument.base}{instrument.term}"  # Bybit format: BTCUSD
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")


def symbol_to_instrument(symbol: str) -> Instrument:
    """
    Parses a symbol like 'BTCUSD', 'FDUSDKRW', 'KRWTRUMP' into a Spot instrument,
    restricting base and term to known Currency enum members.

    Supports:
    - Dash-separated format: 'BTC-USD'
    - Concatenated format: tries longest valid base/term split

    Returns:
        Spot: base and term currencies

    Raises:
        ValueError: if no valid base/term pair is found
    """
    if '-' in symbol:
        parts = symbol.split('-')
        if len(parts) == 2 and parts[0] in Currency.__members__ and parts[1] in Currency.__members__:
            return Spot(base=Currency[parts[0]], term=Currency[parts[1]])
        raise ValueError(f"Invalid dash-separated symbol: {symbol}")

    for split in reversed(range(2, len(symbol) - 1)):
        base, term = symbol[:split], symbol[split:]
        if base in Currency.__members__ and term in Currency.__members__:
            return Spot(base=Currency[base], term=Currency[term])

    raise ValueError(f"Unsupported symbol: {symbol}")

def exchange_symbol_to_instrument_symbol(exchange: Exchange, exchange_symbol: str) -> str:
    """
    Converts an exchange-specific symbol (e.g., 'BTC-USD' or 'BTCUSD') into a normalized instrument.symbol (e.g., 'BTCUSD').

    Args:
        exchange (Exchange): The exchange that produced the symbol.
        exchange_symbol (str): Symbol in the exchange's format.

    Returns:
        str: Normalized instrument.symbol

    Raises:
        ValueError: If parsing fails or exchange is unsupported.
    """
    # Step 1: parse to Spot instrument
    if exchange in {Exchange.COINBASE, Exchange.OKX}:
        if '-' not in exchange_symbol:
            raise ValueError(f"Expected dash in symbol for {exchange}: {exchange_symbol}")
    elif exchange in {Exchange.BINANCE, Exchange.BYBIT}:
        if '-' in exchange_symbol:
            raise ValueError(f"Unexpected dash in symbol for {exchange}: {exchange_symbol}")
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

    instrument = symbol_to_instrument(exchange_symbol)
    return instrument.symbol



def iso_to_epoch_ms(iso_ts: str) -> int:
    """
    Convert ISO 8601 timestamp with nanoseconds (e.g., '2025-05-19 08:44:49.187636535Z')
    to epoch milliseconds.
    """
    if iso_ts.endswith('Z'):
        iso_ts = iso_ts[:-1]

    # Fast truncation: keep only microseconds (first 6 digits)
    if '.' in iso_ts:
        date_part, frac_part = iso_ts.split('.')
        iso_ts = f"{date_part}.{frac_part[:6]}"
    else:
        iso_ts += '.000000'  # ensure microseconds

    # Use strptime for speed + direct UTC awareness
    dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def now_epoch_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def epoch_ms_to_datetime(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

# def epoch_ms_to_iso(epoch_ms: int) -> str:
#     return epoch_ms_to_datetime(epoch_ms).isoformat()


