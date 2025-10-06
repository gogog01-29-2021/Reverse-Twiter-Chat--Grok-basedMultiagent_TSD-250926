import unittest
from datetime import datetime, timedelta

import csp
import numpy as np
from csp import ts, stats


@csp.node
def time_moving_average(x: ts[float], window_size: float) -> ts[float]:
    with csp.start():
        assert window_size > 0
        csp.set_buffering_policy(x, tick_history=timedelta(seconds=window_size))

    if csp.ticked(x):
        buffer = csp.values_at(
            x,
            start_index_or_time=timedelta(seconds=-window_size),
            end_index_or_time=timedelta(seconds=0),
            start_index_policy=csp.TimeIndexPolicy.EXCLUSIVE,
            end_index_policy=csp.TimeIndexPolicy.INCLUSIVE
        )
        if len(buffer) > 0:
            return np.mean(buffer)


@csp.node
def tick_moving_average(x: ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        assert tick_count > 0
        csp.set_buffering_policy(x, tick_count=tick_count)

    if csp.ticked(x):
        buffer = csp.values_at(
            x,
            start_index_or_time=-(tick_count - 1),
            end_index_or_time=0
        )
        if len(buffer) > 0:
            return np.mean(buffer)


class TestTimeMovingAverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.st = datetime(2025, 1, 1)
        cls.et = cls.st + timedelta(seconds=5)
        cls.window_size = 2.0
        cls.price = csp.curve(
            typ=float,
            data=[
                (cls.st + timedelta(seconds=i), val) for i, val in [
                    (1.0, 12.5),
                    (1.5, 12.0),
                    (2.0, 10.0),
                    (2.5, 15.5),
                    (3.0, 11.5),
                    (3.5, 12.0),
                    (4.0, 13.0),
                    (4.5, 11.0),
                    (5.0, 12.5),
                ]
            ]
        )
        cls.expected = [
            (datetime(2025, 1, 1, 0, 0, 1), 12.5),
            (datetime(2025, 1, 1, 0, 0, 1, 500000), 12.25),
            (datetime(2025, 1, 1, 0, 0, 2), 11.5),
            (datetime(2025, 1, 1, 0, 0, 2, 500000), 12.5),
            (datetime(2025, 1, 1, 0, 0, 3), 12.25),
            (datetime(2025, 1, 1, 0, 0, 3, 500000), 12.25),
            (datetime(2025, 1, 1, 0, 0, 4), 13.0),
            (datetime(2025, 1, 1, 0, 0, 4, 500000), 11.875),
            (datetime(2025, 1, 1, 0, 0, 5), 12.125)
        ]

    def test_time_moving_average(self):
        @csp.graph
        def graph():
            node = time_moving_average(x=self.price, window_size=self.window_size)
            csp.add_graph_output("data", node)

        result = csp.run(graph, starttime=self.st, endtime=self.et)
        self.assertListEqual(result["data"], self.expected)

    def test_csp_stats(self):
        @csp.graph
        def graph():
            node = stats.mean(
                self.price,
                interval=timedelta(seconds=self.window_size),
                min_window=timedelta(seconds=0),
                min_data_points=0
            )
            csp.add_graph_output("data", node)

        result = csp.run(graph, starttime=self.st, endtime=self.et)
        self.assertListEqual(result["data"], self.expected)


if __name__ == "__main__":
    unittest.main()
