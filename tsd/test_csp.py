#!/usr/bin/env python3
"""
Test script to verify CSP installation and basic functionality
"""

import csp
from datetime import datetime, timedelta

# Test 1: Basic csp.Struct functionality
print("=== Testing csp.Struct ===")

class MyData(csp.Struct):
    a: int
    b: str = 'default'
    c: list

# Create instance
data = MyData(a=42, c=[1, 2, 3])
print(f"MyData instance: {data}")
print(f"Field a: {data.a}")
print(f"Field b: {data.b}")
print(f"Field c: {data.c}")

# Test 2: List field types
print("\n=== Testing List Field Types ===")

# Python list field
class A(csp.Struct):
    a: list

s = A(a=[1, 'x'])
s.a.append(True)
print(f"Using Python list field: value {s.a}, type {type(s.a)}, is Python list: {isinstance(s.a, list)}")

# Test 3: Basic CSP graph
print("\n=== Testing Basic CSP Graph ===")

class Trade(csp.Struct):
    price: float
    size: int

@csp.graph
def test_graph():
    trades = csp.curve(Trade,
                       [(datetime(2020, 1, 1), Trade(price=100.01, size=200)),
                        (datetime(2020, 1, 1, 0, 0, 1), Trade(price=100.02, size=300))]
             )

    sizes = trades.size
    cumqty = csp.accum(sizes)

    csp.print('trades', trades)
    csp.print('cumqty', cumqty)

try:
    print("Running CSP graph...")
    csp.run(test_graph, starttime=datetime(2020, 1, 1))
    print("CSP graph completed successfully!")
except Exception as e:
    print(f"Error running CSP graph: {e}")

print("\n=== CSP Test Complete ===")
