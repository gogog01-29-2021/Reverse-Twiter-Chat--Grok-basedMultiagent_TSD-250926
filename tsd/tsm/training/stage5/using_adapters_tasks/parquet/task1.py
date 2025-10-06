import logging
import tempfile
from datetime import datetime, timedelta

import csp
from csp.adapters.parquet import ParquetOutputConfig, ParquetReader, ParquetWriter


class MarketTick(csp.Struct):
    """Represents a market tick with trade volume and price."""
    trade_volume: int
    trade_price: float


@csp.graph
def write_market_ticks(file_name: str):
    """Writes a series of market tick data into a Parquet file."""
    start_time = datetime(2020, 1, 1)

    ticks = [
        (start_time + timedelta(seconds=1), MarketTick(trade_volume=100, trade_price=100.25)),
        (start_time + timedelta(seconds=2), MarketTick(trade_volume=150, trade_price=101.0)),
        (start_time + timedelta(seconds=3), MarketTick(trade_volume=120, trade_price=102.5)),
    ]

    tick_curve = csp.curve(MarketTick, ticks)

    writer = ParquetWriter(
        file_name=file_name,
        timestamp_column_name="timestamp",
        config=ParquetOutputConfig(allow_overwrite=True),
    )
    writer.publish_struct(tick_curve)



@csp.graph
def write_market_series(file_name: str):
    """Writes market volume and price series separately into a Parquet file."""
    start_time = datetime(2020, 1, 1)

    volume_curve = csp.curve(int, [(start_time + timedelta(seconds=i), 100 + i * 10) for i in range(3)])
    price_curve = csp.curve(float, [(start_time + timedelta(seconds=i), 100.0 + i * 0.75) for i in range(3)])

    writer = ParquetWriter(
        file_name=file_name,
        timestamp_column_name="timestamp",
        config=ParquetOutputConfig(allow_overwrite=True),
    )
    writer.publish("trade_volume", volume_curve)
    writer.publish("trade_price", price_curve)


@csp.graph
def market_data_writer(struct_file: str, series_file: str):
    """Runs both market tick and series writers."""
    write_market_ticks(struct_file)
    write_market_series(series_file)


@csp.graph
def read_market_data(struct_file: str, series_file: str):
    """Reads market data from Parquet files and prints it."""
    tick_reader = ParquetReader(struct_file, time_column="timestamp")
    tick_data = tick_reader.subscribe_all(MarketTick)
    csp.print("MarketTick Data", tick_data)

    series_reader = ParquetReader(series_file, time_column="timestamp")
    series_data = series_reader.subscribe_all(MarketTick)
    csp.print("Market Series Data", series_data)


def main():
    with tempfile.NamedTemporaryFile(suffix=".parquet") as struct_file:
        struct_file.file.close()
        with tempfile.NamedTemporaryFile(suffix=".parquet") as series_file:
            series_file.file.close()

            # Write market data
            csp.run(
                market_data_writer,
                struct_file.name,
                series_file.name,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(minutes=1),
            )

            print("\n--- Market Data Written to Parquet ---\n")

            # Read market data
            csp.run(
                read_market_data,
                struct_file.name,
                series_file.name,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(minutes=1),
            )

            try:
                import pandas
            except ModuleNotFoundError:
                logging.warning("Pandas is not installed. Unable to display dataframes.")
            else:
                tick_df = pandas.read_parquet(struct_file.name)
                print(f"\nMarketTick DataFrame:\n{tick_df}")

                series_df = pandas.read_parquet(series_file.name)
                print(f"\nMarket Series DataFrame:\n{series_df}")


if __name__ == "__main__":
    main()
