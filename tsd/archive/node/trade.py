from datetime import datetime, timedelta

import csp
from csp import ts

from core.types.instrument import Instrument
from core.types.trade import Trade, OrderSide

# TODO: requires refactoring


@csp.node
def vwap(trade: ts[Trade]) -> ts[float]:
    with csp.state():
        s_cum_notional = 0.
        s_cum_qty = 0.

    if csp.ticked(trade):
        s_cum_notional += trade.price * trade.qty
        s_cum_qty += trade.qty

        return s_cum_notional / s_cum_qty


@csp.graph
def test_vwap():
    st = datetime.now()

    instr = Instrument(name="Dummy")

    trades = csp.curve(
        Trade,
        [
            (st + timedelta(seconds=1), Trade(instr=instr, price=100.0, qty=200, side=OrderSide.BUY)),
            (st + timedelta(seconds=2), Trade(instr=instr, price=101.5, qty=500, side=OrderSide.SELL)),
            (st + timedelta(seconds=3), Trade(instr=instr, price=100.5, qty=100, side=OrderSide.BUY)),
            (st + timedelta(seconds=4), Trade(instr=instr, price=101.2, qty=500, side=OrderSide.SELL)),
            (st + timedelta(seconds=5), Trade(instr=instr, price=101.3, qty=500, side=OrderSide.SELL)),
            (st + timedelta(seconds=6), Trade(instr=instr, price=101.4, qty=500, side=OrderSide.BUY)),
        ]
    )

    csp.split()
