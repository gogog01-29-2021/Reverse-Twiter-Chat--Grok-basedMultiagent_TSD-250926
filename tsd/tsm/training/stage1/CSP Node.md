## Table of Contents

- [Table of Contents](#table-of-contents)
- [CSP](#csp)
  - [Real-time event stream processing](#real-time-event-stream-processing)
  - [Imports and time series](#imports-and-time-series)
- [What is a CSP Node?](#what-is-a-csp-node)
- [Anatomy of a `csp.node`](#anatomy-of-a-cspnode)
  - [Run the program](#run-the-program)
- [Basket inputs](#basket-inputs)
- [Node Outputs](#node-outputs)
- [Basket Outputs](#basket-outputs)
- [Generic Types](#generic-types)

## CSP

`csp` is a stream processing library that integrates real-time and historical playback workflows. Users compose a directed acyclic graph (DAG) of transformations and run them as a single pipeline with multiple data sources.

### Real-time event stream processing

Real-time data is continuously updated in a stream of *events* that occur at unpredictable times. Stream processors handle these events and analyze their data so that underlying insights can be achieved in real-time. Each application is driven by updates to the input source values which propagate down the data pipeline. In `csp`, we refer to the input changes as "ticks" and write the analysis workflow as a directed graph.

`csp` programs are written in a functional-style, and consist of:

- runtime components in the form of `csp.node` methods, and
- graph-building components in the form of `csp.graph` methods.

### Imports and time series

```python
from datetime import datetime, timedelta

import csp
from csp import ts
```

`csp` defines a high-level `ts` type that denotes a time series input. The `ts` is to `csp` as the `Series` is to `pandas` or the `array` is to `numpy`: it is the fundamental data unit. A time series is a single stream of typed values. Whenever the stream is updated, any *consumer* of that data is invoked. Consumers are nodes or output adapters that ingest time series data.

When you access a time series within a *node*, you access the *last value* of the time series. You can also store a fixed or variable sized history for any input time series.
When you access a time series within a *graph*, you access the *edge* of the time series. This is simply the type, input and output definition of the data; it is used for building the graph at startup and ensuring type validation.


## What is a CSP Node?

A csp.node is a fundamental unit of computation in CSP. It defines what to compute each time new data (a "tick") arrives.

Why do we need it? In real-time systems, we need to respond instantly to incoming data. A node allows us to define how to react to these events.


---

## Anatomy of a `csp.node`

The heart of a calculation graph are the `csp.nodes` that run the computations.
`csp.node` methods can take any number of scalar and timeseries arguments, and can return 0 → N timeseries outputs.
Timeseries inputs/outputs should be thought of as the edges that connect components of the graph.
These "edges" can tick whenever they have a new value.
Every tick is associated with a value and the time of the tick.
`csp.nodes` can have various other features, here is an example of a `csp.node` that demonstrates many of the features.
Keep in mind that nodes will execute repeatedly as inputs tick with new data.
They may (or may not) generate an output as a result of an input tick.

```python
from datetime import datetime, timedelta

import csp
from csp import ts

@csp.node(name='my_node')                                                  # 1 
def mid_price_node(n: int, bid: ts[float], ask: ts[float]) -> ts[float]:   # 2
    with csp.alarms():                                                     # 3
        # Define an alarm time-series of type bool                         # 4
        alarm = csp.alarm(bool)                                            # 5
                                                                           # 6
    with csp.state():                                                      # 7
        # Create a state variable bound to the node                        # 8
        s_mid = 0.0                                                        # 9
                                                                           # 10
    with csp.start():                                                      # 11
        # Code block that executes once on start of the engine             # 12
        # one can set timeseries properties here as well, such as          # 13
        # csp.set_buffering_policy(bid, tick_count=5)                      # 14
        # csp.set_buffering_policy(bid, tick_history=timedelta(minutes=1)) # 15
        # csp.make_passive(bid)                                            # 16
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)              # 17
                                                                           # 18
    with csp.stop():                                                       # 19
        pass  # code block to execute when the engine is done              # 20
                                                                           # 21
    if csp.ticked(bid, ask) and csp.valid(bid, ask):                       # 22
        s_mid = (bid + ask) / 2.0                                          # 23
                                                                           # 24
    if csp.ticked(alarm):                                                  # 25
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)              # 26
        return s_mid                                                       # 27
```

Let's review line by line

1\) Every CSP node must start with the **`@csp.node`** decorator. The name of the node will be the name of the function, unless a `name` argument is provided. The name is used when visualizing a graph with `csp.show_graph` or profiling with CSP's builtin [`profiler`](#Profile-csp-code).

2\) CSP nodes are fully typed and type-checking is strictly enforced by the C++ engine. All arguments must be typed, as well as all outputs. Outputs are typed using function annotation syntax.

Single outputs can be unnamed, for multiple outputs they must be named.
When using multiple outputs, annotate the type using **`def my_node(inputs) → csp.Outputs(name1=ts[<T>], name2=ts[<V>])`** where `T` and `V` are the respective types of `name1` and `name2`.

Note the syntax of timeseries inputs, they are denoted by **`ts[type]`**.
Scalars can be passed in as regular types, in this example we pass in `n` which expects a type of `int`

3\) **`with csp.alarms()`**: nodes can (optionally) declare internal alarms, every instance of the node will get its own alarm that can be scheduled and act just like a timeseries input.
All alarms must be declared within the alarms context.

5\) Instantiate an alarm in the alarms context using the `csp.alarm(type)` function. This creates an alarm which is a time-series of type `type`.

7\) **`with csp.state()`**: optional state variables can be defined under the state context.
Note that variables declared in state will live across invocations of the method.

9\) An example declaration and initialization of state variable `s_mid`.
It is good practice to name state variables prefixed with `s_`, which is the convention in the CSP codebase.

11\) **`with csp.start()`**: an optional block to execute code at the start of the engine.
Generally this is used to setup initial timers or set input timeseries properties such as buffer sizes, or to make inputs passive

14-15) **`csp.set_buffering_policy`**: nodes can request a certain amount of history be kept on the incoming time series, this can be denoted in number of ticks or in time.
By setting a buffering policy, nodes can access historical values of the timeseries (by default only the last value is kept)

16\) **`csp.make_passive`** / **`csp.make_active`**: Nodes may not need to react to all of their inputs, they may just need their latest value.
For performance purposes the node can mark an input as passive to avoid triggering the node unnecessarily.
`make_active` can be called to reactivate an input.

17\) **`csp.schedule_alarm`**: scheduled a one-shot tick on the given alarm input.
The values given are the timedelta before the alarm triggers and the value it will have when it triggers.
Note that `schedule_alarm` can be called multiple times on the same alarm to schedule multiple triggers.

19\) **`with csp.stop()`** is an optional block that can be called when the engine is done running.

22\) All nodes will have if conditions to react to different inputs.
**`csp.ticked()`** takes any number of inputs and returns true if **any** of the inputs ticked.
**`csp.valid()`** similar takes any number of inputs however it only returns true if **all** inputs are valid.
Valid means that an input has had at least one tick and so it has a "current value".

23\) One of the benefits of CSP is that you always have easy access to the latest value of all inputs.
`bid` and `ask` on line 22, 23 will always have the latest value of both inputs, even if only one of them just ticked.

25\) This demonstrates how an alarm can be treated like any other input.

27\) We tick our running "mid" as an output here every second.

### Run the program

Note: The concepts of csp.graph and csp.curve will be covered in detail in the next stage. For now, feel free to skim through this section, and refer to the [stage2](../stage2/CSP%20Graph.md) if you're curious to learn more.

To execute a `csp` application we use the `run` function. Each run starts at a `starttime` and ends at an `endtime`. For the example above, we can run the graph using:

We can run the node using **`csp.run`** as follows

```python
@csp.graph
def my_graph(start: datetime):
    # Example time-series for bid and ask (dynamic data)
    bid = csp.curve(typ=float, data=[
        (start, 100.5),
        (start + timedelta(seconds=1), 101.0),
        (start + timedelta(seconds=2), 101.5),
        (start + timedelta(seconds=3), 102.0),
        (start + timedelta(seconds=4), 102.5),
    ])

    ask = csp.curve(typ=float, data=[
        (start, 101.5),
        (start + timedelta(seconds=1), 102.0),
        (start + timedelta(seconds=2), 102.5),
        (start + timedelta(seconds=3), 103.0),
        (start + timedelta(seconds=4), 103.5),
    ])
    # Run demo_node
    mid = mid_price_node(1, bid, ask)

    # Print results
    csp.print("bid", bid)
    csp.print("ask", ask)
    csp.print("mid", mid)

# Run the graph
def main():
    start = datetime(2025, 3, 17, 9, 30, 0)

    csp.run(my_graph, start, starttime=start, endtime=timedelta(seconds=4))

if __name__ == "__main__":
    main()
```

The program will produce the following:

```python-console
2025-03-17 09:30:00 bid:100.5
2025-03-17 09:30:00 ask:101.5
2025-03-17 09:30:01 bid:101.0
2025-03-17 09:30:01 ask:102.0
2025-03-17 09:30:01 mid:101.5
2025-03-17 09:30:02 bid:101.5
2025-03-17 09:30:02 ask:102.5
2025-03-17 09:30:02 mid:102.0
2025-03-17 09:30:03 bid:102.0
2025-03-17 09:30:03 ask:103.0
2025-03-17 09:30:03 mid:102.5
2025-03-17 09:30:04 bid:102.5
```

Take a moment to reason with this output. Each `mid_price_node` node is invoked when `alarm` ticks.



## Basket inputs

In addition to single time-series inputs, a node can also accept a **basket** of time series as an argument.
A basket is essentially a collection of timeseries which can be passed in as a single argument.
Baskets can either be list baskets or dict baskets.
Individual timeseries in a basket can tick independently, and they can be looked at and reacted to individually or as a collection.

For example:

```python
@csp.node                                      # 1
def demo_basket_node(                          # 2
    list_basket: [ts[int]],                    # 3
    dict_basket: {str: ts[int]}                # 4
) -> ts[float]:                                # 5
                                               # 6
    if csp.ticked(list_basket):                # 7
        return sum(list_basket.validvalues())  # 8
                                               # 9
    if csp.ticked(list_basket[3]):             # 10
        return list_basket[3]                  # 11
                                               # 12
    if csp.ticked(dict_basket):                # 13
        # can iterate over ticked key,items    # 14
        # for k,v in dict_basket.tickeditems():# 15
        #     ...                              # 16
        return sum(dict_basket.tickedvalues()) # 17
```

3\) Note the syntax of basket inputs.
list baskets are noted as `[ts[type]]` (a list of time series) and dict baskets are `{key_type: ts[ts_type]}` (a dictionary of timeseries keyed by type `key_type`). It is also possible to use the `List[ts[int]]` and `Dict[str, ts[int]]` typing notation.

7\) Just like single timeseries, we can react to a basket if it ticked.
The convention is the same as passing multiple inputs to `csp.ticked`, `csp.ticked` is true if **any** basket input ticked.
`csp.valid` is true is **all** basket inputs are valid.

8\) baskets have various iterators to access their inputs:

- **`tickedvalues`**: iterator of values of all ticked inputs
- **`tickedkeys`**: iterator of keys of all ticked inputs (keys are list index for list baskets)
- **`tickeditems`**: iterator of (key,value) tuples of ticked inputs
- **`validvalues`**: iterator of values of all valid inputs
- **`validkeys`**: iterator of keys of all valid inputs
- **`validitems`**: iterator of (key,value) tuples of valid inputs
- **`keys`**: list of keys on the basket (**dictionary baskets only** )

10-11) This demonstrates the ability to access an individual element of a
basket and react to it as well as access its current value

## **Node Outputs**

Nodes can return any number of outputs (including no outputs, in which case it is considered an "output" or sink node).
Nodes with single outputs can return the output as an unnamed output.
Nodes returning multiple outputs must have them be named.
When a node is called at graph building time, if it is a single unnamed node the return variable is an edge representing the output which can be passed into other nodes.
An output timeseries cannot be ticked more than once in a given node invocation.
If the outputs are named, the return value is an object with the outputs available as attributes.
For example (examples below demonstrate various ways to output the data as well)

```python
@csp.node
def single_unnamed_outputs(n: ts[int]) -> ts[int]:
    # can either do
    return n
    # or
    # csp.output(n) to continue processes after output


@csp.node
def multiple_named_outputs(n: ts[int]) -> csp.Outputs(y=ts[int], z=ts[float]):
    # can do
    # csp.output(y=n, z=n+1.) to output to multiple outputs
    # or separate the outputs to tick out at separate points:
    # csp.output(y=n)
    # ...
    # csp.output(z=n+1.)
    # or can return multiple values with:
    return csp.output(y=n, z=n+1.)

@csp.graph
def my_graph(n: ts[int]):
    x = single_unnamed_outputs(n)
    # x represents the output edge of single_unnamed_outputs,
    # we can pass it a time series input to other nodes
    csp.print('x', x)


    result = multiple_named_outputs(n)
    # result holds all the outputs of multiple_named_outputs, which can be accessed as attributes
    csp.print('y', result.y)
    csp.print('z', result.z)
```

## Basket Outputs

Similarly to inputs, a node can also produce a basket of timeseries as an output.
For example:

```python
class MyStruct(csp.Struct):                                               # 1
    symbol: str                                                           # 2
    index: int                                                            # 3
    value: float                                                          # 4
                                                                          # 5
@csp.node                                                                 # 6
def demo_basket_output_node(                                              # 7
    in_: ts[MyStruct],                                                    # 8
    symbols: [str],                                                       # 9
    num_symbols: int                                                      # 10
) -> csp.Outputs(                                                         # 11
    dict_basket=csp.OutputBasket({str: ts[float]}, shape="symbols"),  # 15
    list_basket=csp.OutputBasket([ts[float]], shape="num_symbols"),   # 16
):                                                                        # 17
                                                                          # 18
    if csp.ticked(in_):                                                   # 19
        # output to dict basket                                           # 20
        csp.output(dict_basket[in_.symbol], in_.value)                    # 21
        # alternate output syntax, can output multiple keys at once       # 22
        # csp.output(dict_basket={in_.symbol: in_.value})                 # 23
        # output to list basket                                           # 24
        csp.output(list_basket[in_.index], in_.value)                     # 25
        # alternate output syntax, can output multiple keys at once       # 26
        # csp.output(list_basket={in_.index: in_.value})                  # 27
```

11-17) Note the output declaration syntax.
A basket output can be either named or unnamed (both examples here are named), and its shape can be specified two ways.
The `shape` parameter is used with a scalar value that defines the shape of the basket, or the name of the scalar argument (a dict basket expects shape to be a list of keys. lists basket expects `shape` to be an `int`).
`shape_of` is used to take the shape of an input basket and apply it to the output basket.

20+) There are several choices for output syntax.
The following work for both list and dict baskets:

- `csp.output(basket={key: value, key2: value2, ...})`
- `csp.output(basket[key], value)`
- `csp.output({key: value}) # only works if the basket is the only output`

## Generic Types

CSP supports syntax for generic types as well.
To denote a generic type we use a string (typically `'T'` is used) to denote a generic type.
When a node is called the type of the argument will get bound to the given type variable, and further inputs / outputs will be checked and bound to said typevar.
Note that the string syntax `'~T'` denotes the argument expects the *value* of a type, rather than a type itself:

```python
@csp.node
def sample(trigger: ts[object], x: ts['T']) -> ts['T']:
    '''will return current value of x on trigger ticks'''
    with csp.state():
        csp.make_passive(x)

    if csp.ticked(trigger) and csp.valid(x):
        return x


@csp.node
def const(value: '~T') -> ts['T']:
    ...
```

`sample` takes a timeseries of type `'T'` as an input, and returns a timeseries of type `'T'`.
This allows us to pass in a `ts[int]` for example, and get a `ts[int]` as an output, or `ts[bool]` → `ts[bool]`

`const` takes value as an *instance* of type `T`, and returns a timeseries of type `T`.
So we can call `const(5)` and get a `ts[int]` output, or `const('hello!')` and get a `ts[str]` output, etc...

If a value is provided rather than an explicit type argument (for example, to `const`) then CSP resolves the type using internal logic. In some cases, it may be easier to override the automatic type inference.
Users can force a type variable to be a specific value with the `.using` function. For example, `csp.const(1)` will be resolved to a `ts[int]`; if you want to instead force the type to be `float`, do `csp.const.using(T=float)(1)`.