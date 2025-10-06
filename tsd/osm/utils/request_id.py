import threading

class RequestIDGenerator:
    def __init__(self):
        self._lock = threading.Lock()
        self._counter = 1

    def next_id(self) -> int:
        with self._lock:
            val = self._counter
            self._counter += 1
            return val


request_id_generator = RequestIDGenerator()

def get_next_request_id() -> int:
    return request_id_generator.next_id()