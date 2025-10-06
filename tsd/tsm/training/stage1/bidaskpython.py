import time
from datetime import datetime, timedelta
from typing import List, Tuple, Callable, Any
import threading
from queue import Queue

class PurePythonStream:
    def __init__(self):
        self.subscribers: List[Callable] = []
        self.data: List[Tuple[datetime, Any]] = []
    
    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)
    
    def emit(self, timestamp: datetime, value: Any):
        self.data.append((timestamp, value))
        for callback in self.subscribers:
            callback(timestamp, value)

class StreamProcessor:
    def __init__(self):
        self.streams = {}
        self.event_queue = Queue()
        
    def create_stream(self, name: str, data: List[Tuple[datetime, Any]]):
        stream = PurePythonStream()
        self.streams[name] = stream
        
        # Schedule events
        for timestamp, value in data:
            self.event_queue.put((timestamp, name, value))
        
        return stream
    
    def run(self, start_time: datetime, end_time: datetime):
        current_time = start_time
        
        # Sort events by time
        events = []
        while not self.event_queue.empty():
            events.append(self.event_queue.get())
        events.sort(key=lambda x: x[0])
        
        # Process events in time order
        for timestamp, stream_name, value in events:
            if start_time <= timestamp <= start_time + end_time:
                print(f"{timestamp} {stream_name}: {value}")
                self.streams[stream_name].emit(timestamp, value)

def pure_python_bid_ask():
    processor = StreamProcessor()
    start = datetime(2020, 1, 1)
    
    # Create bid stream
    bid_data = [
        (start, 100.0),
        (start + timedelta(seconds=1), 100.1),
        (start + timedelta(seconds=2), 100.2),
    ]
    
    # Create ask stream  
    ask_data = [
        (start, 100.5),
        (start + timedelta(seconds=1), 100.6),
        (start + timedelta(seconds=2), 100.7),
    ]
    
    bid_stream = processor.create_stream("BID", bid_data)
    ask_stream = processor.create_stream("ASK", ask_data)
    
    # Calculate spread when both bid and ask are available
    latest_bid = None
    latest_ask = None
    
    def on_bid(timestamp, value):
        nonlocal latest_bid
        latest_bid = (timestamp, value)
        if latest_ask and latest_ask[0] == timestamp:
            spread = latest_ask[1] - value
            print(f"{timestamp} SPREAD: {spread}")
    
    def on_ask(timestamp, value):
        nonlocal latest_ask
        latest_ask = (timestamp, value)
        if latest_bid and latest_bid[0] == timestamp:
            spread = value - latest_bid[1]
            print(f"{timestamp} SPREAD: {spread}")
    
    bid_stream.subscribe(on_bid)
    ask_stream.subscribe(on_ask)
    
    processor.run(start, timedelta(seconds=5))

if __name__ == "__main__":
    pure_python_bid_ask()