from datetime import datetime, timedelta

import csp
from csp import ts

# TODO: requires refactoring


@csp.node
def sum_node(sources: [ts[float]]) -> ts[float]:
    if (csp.ticked(sources)):
        return sum(sources.validvalues())


@csp.node
def cum_sum_node(x: ts[float]) -> ts[float]:
    with csp.state():
        s_cum_sum = 0
    if csp.ticked(x):
        s_cum_sum += x
        return s_cum_sum


@csp.graph
def test_sum_node():
    x = csp.const(1)
    y = csp.const(2)
    z = csp.const(3)

    num_sum = sum_node([x, y, z])

    csp.print("x", x)
    csp.print("y", y)
    csp.print("z", z)
    csp.print("sum", num_sum)


@csp.graph
def test_cum_sum_node():
    st = datetime.now()
    x = csp.curve(
        float,
        [
            (st + timedelta(1), 1),
            (st + timedelta(2), 2),
            (st + timedelta(3), 3),
            (st + timedelta(3), 4),
            (st + timedelta(3), 5)
        ]
    )
    cum_sum = cum_sum_node(x)
    csp.print("x", x)
    csp.print("cum_sum", cum_sum)


def main():
    csp.run(test_sum_node, starttime=datetime.now())
    csp.run(test_cum_sum_node, starttime=datetime.now())


if __name__ == "__main__":
    main()
