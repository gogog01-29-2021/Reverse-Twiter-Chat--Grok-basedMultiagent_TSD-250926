import csp

from core.types.instrument import Instrument, Currency, Spot


class Position(csp.Struct):
    instr: Instrument
    qty: float
    time_received: int

    def __repr__(self):
        return f"Position( instr={self.instr.symbol}, qty={self.qty}, time_received={self.time_received} )"

    def __str__(self):
        return f"Position( instr={self.instr.symbol}, qty={self.qty}, time_received={self.time_received} )"


if __name__ == "__main__":
    pos = Position(instr=Spot(base=Currency.BTC, term=Currency.USD), qty=3, time_received=0)
    print(pos)
