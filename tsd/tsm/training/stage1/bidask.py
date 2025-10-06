import csp
from datetime import datetime, timedelta

@csp.graph
def simple_bid_ask_example():
    # Create some fake bid/ask data
    start = datetime(2020, 1, 1)
    
    # Bid prices (what buyers are willing to pay)
    bid_data = csp.curve(typ=float, data=[
        (start, 100.0),
        (start + timedelta(seconds=1), 100.1),
        (start + timedelta(seconds=2), 100.2),
    ])
    
    # Ask prices (what sellers want)
    ask_data = csp.curve(typ=float, data=[
        (start, 100.5),
        (start + timedelta(seconds=1), 100.6),
        (start + timedelta(seconds=2), 100.7),
    ])
    
    # Print the values so we can see them
    csp.print("BID", bid_data)
    csp.print("ASK", ask_data)

# Run the graph
if __name__ == "__main__":
    start_time = datetime(2020, 1, 1)
    csp.run(simple_bid_ask_example, 
            starttime=start_time, 
            endtime=timedelta(seconds=5))