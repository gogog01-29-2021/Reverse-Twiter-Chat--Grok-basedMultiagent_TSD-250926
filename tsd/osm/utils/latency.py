import time
import functools

def measure_latency(name: str = "operation"):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            result = await func(*args, **kwargs)
            end = time.perf_counter_ns()
            latency_ms = (end - start) / 1_000_000
            print(f"[Latency] {name}: {latency_ms:.3f} ms")
            return result
        return wrapper
    return decorator