The CSP engine can be run in two flavors, realtime and simulation.

In simulation mode, the engine is always run at full speed pulling in time-based data from its input adapters and running them through the graph.
All inputs in simulation are driven off the provided timestamped data of its inputs.

In realtime mode, the engine runs in wallclock time as of "now".
Realtime engines can get data from realtime adapters which source data on separate threads and pass them through to the engine (ie think of activeMQ events happening on an activeMQ thread and being passed along to the engine in "realtime").

Since engines can run in both simulated and realtime mode, users should **always** use **`csp.now()`** to get the current time in a `csp.node`.

## Table of Contents

- [Understand how CSP runs](#understand-how-csp-runs)
  - [Simulation Mode](#simulation-mode)
  - [Realtime Mode](#realtime-mode)
  - [csp.PushMode](#csppushmode)
  - [Handling duplicate timestamps](#handling-duplicate-timestamps)
  - [`csp.unroll`](#cspunroll)
  - [`csp.feedback`](#cspfeedback)
  - [Realtime Group Event Synchronization](#realtime-group-event-synchronization)
- [Performance Analysis with Profiling](#performance-analysis-with-profiling)
  - [Profiling a real-time `csp.graph`](#profiling-a-real-time-cspgraph)
  - [Saving raw profiling data to a file](#saving-raw-profiling-data-to-a-file)
  - [graph\_info: build-time information](#graph_info-build-time-information)

# Understand how CSP runs

## Simulation Mode

Simulation mode is the default mode of the engine.
As stated above, simulation mode is used when you want your engine to crunch through historical data as fast as possible.
In simulation mode, the engine runs on some historical data that is fed in through various adapters.
The adapters provide events by time, and they are streamed into the engine via the adapter timeseries in time order.
`csp.timer` and `csp.node` alarms are scheduled and executed in "historical time" as well.
Note that there is no strict requirement for simulated runs to run on historical dates.
As long as the engine is not in realtime mode, it remains in simulation mode until the provided endtime, even if endtime is in the future.

## Realtime Mode

   wallclock "now" as of the time of calling run.
Once the simulation run is done, the engine switches into realtime mode.
Under realtime mode, external realtime adapters will be able to send data into the engine thread.
All time based inputs such as `csp.timer` and alarms will switch to executing in wallclock time as well.

As always, `csp.now()` should still be used in `csp.node` code, even when running in realtime mode.
`csp.now()` will be the time assigned to the current engine cycle.

## csp.PushMode

When consuming data from input adapters there are three choices on how one can consume the data:

| PushMode           | EngineMode | Description                                                                                                                                                       |
| :----------------- | :--------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LAST_VALUE**     | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once with the last value on a given timestamp                            |
|                    | Realtime   | all ticks that occurred since previous engine cycle will collapse / conflate to the latest value                                                                  |
| **NON_COLLAPSING** | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once per engine cycle. subsequent cycles will execute with the same time |
|                    | Realtime   | all ticks that occurred since previous engine cycle will be ticked across subsequent engine cycles as fast as possible                                            |
| **BURST**          | Simulation | all ticks from input source with duplicate timestamps (on the same timeseries) will tick once with a list of all values                                           |
|                    | Realtime   | all ticks that occurred since previous engine cycle will tick once with a list of all the values                                                                  |

## Handling duplicate timestamps

In `csp`, there can be multiple engine cycles that occur at the same engine time. This is often the case when using nodes with internal alarms (e.g. [`csp.unroll`](../stage4/Execution-Modes.md)) or using feedback edges ([`csp.feedback`](../stage4/Execution-Modes.md)).


## `csp.unroll`

```python
csp.unroll(x: ts[['T']]) → ts['T']
```

Given a timeseries of a *list* of values, unroll will "unroll" the values in the list into a timeseries of the elements.
`unroll` will ensure to preserve the order across all list ticks.
Ticks will be unrolled in subsequent engine cycles.
For a detailed explanation of this behavior, see the documentation on [duplicate timestamps](Execution-Modes#handling-duplicate-timestamps).


## `csp.feedback`

```python
csp.feedback(typ)
```

`csp.feedback` is a construct that can be used to create artificial loops in the graph.
Use feedbacks in order to delay bind an input to a node in order to be able to create a loop
(think of writing a simulated exchange that takes orders in and needs to feed responses back to the originating node).

`csp.feedback` itself is not an edge, its a construct that allows you to access the delayed edge / bind a delayed input.

Args:

- **`typ`**: type of the edge's data to be bound

Methods:

- **`out()`**: call this method on the feedback object to get the edge which can be wired as an input
- **`bind(x: ts[object])`**: call this to bind an edge to the feedback


If multiple events are scheduled at the same timestamp on a single time-series edge, they will be executed on separate cycles *in the order* they were scheduled. For example, consider the code snippet below:

```python
import csp
from csp import ts
from datetime import datetime, timedelta

@csp.node
def ticks_n_times(x: ts[int], n: int) -> ts[int]:
    # Ticks out a value n times, incrementing it each time
    with csp.alarms():
        alarm = csp.alarm(int)

    if csp.ticked(x):
        for i in range(n):
            csp.schedule_alarm(alarm, timedelta(), x+i)

    if csp.ticked(alarm):
        return alarm

@csp.graph
def duplicate_timestamps():
    v = csp.const(1)
    csp.print('ticks_once', ticks_n_times(v, 1))
    csp.print('ticks_twice', ticks_n_times(v, 2))
    csp.print('ticks_thrice', ticks_n_times(v, 3))

csp.run(duplicate_timestamps, starttime=datetime(2020,1,1))
```

When we run this graph, the output is:

```raw
2020-01-01 00:00:00 ticks_once:1
2020-01-01 00:00:00 ticks_twice:1
2020-01-01 00:00:00 ticks_thrice:1
2020-01-01 00:00:00 ticks_twice:2
2020-01-01 00:00:00 ticks_thrice:2
2020-01-01 00:00:00 ticks_thrice:3
```

A real life example is when using `csp.unroll` to tick out a list of values on separate engine cycles. If we were to use `csp.sample` on the output, we would get the *first* value that is unrolled at each timestamp. Why?
The event that is scheduled on the sampling timer is its first (and only) event at that time; thus, it is executed on the first engine cycle, and samples the first unrolled value.

```python
def sampling_unroll():
    u = csp.unroll(csp.const.using(T=[int])([1, 2, 3]))
    s = csp.sample(csp.const(True), u)
    csp.print('unrolled', u)
    csp.print('sampled', s)
    
csp.run(sampling_unroll, starttime=datetime(2020,1,1))
```

Output:

```raw
2020-01-01 00:00:00 unrolled:1
2020-01-01 00:00:00 sampled:1
2020-01-01 00:00:00 unrolled:2
2020-01-01 00:00:00 unrolled:3
```

## Realtime Group Event Synchronization

The CSP framework supports properly synchronizing events across multiple timeseries that are sourced from the same realtime adapter.
A classical example of this is a market data feed.
Say you consume bid, ask and trade as 3 separate time series for the same product / exchange.
Since the data flows in asynchronously from a separate thread, bid, ask and trade events could end up executing in the engine at arbitrary slices of time, leading to crossed books and trades that are out of range of the bid/ask.
The engine can properly provide a correct synchronous view of all the inputs, regardless of their PushModes.
Its up to adapter implementations to determine which inputs are part of a synchronous "PushGroup".

Here's a classical example.
An Application wants to consume conflating bid/ask as LAST_VALUE but it doesn't want to conflate trades, so its consumed as NON_COLLAPSING.

Lets say we have this sequence of events on the actual market data feed's thread, coming in one the wire in this order.
The columns denote the time the callbacks come in off the market data thread.

<table>
<tbody>
<tr>
<th>Event</th>
<th>T</th>
<th>T+1</th>
<th>T+2</th>
<th>T+3</th>
<th>T+4</th>
<th>T+5</th>
<th>T+6</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.00</td>
<td>100.01</td>
<td><br />
</td>
<td>99.97</td>
<td>99.98</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.02</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
<td><br />
</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td><br />
</td>
<td><br />
</td>
<td>100.02</td>
<td><br />
</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>

Without any synchronization you can end up with nonsensical views based on random timing.
Here's one such possibility (bid/ask are still LAST_VALUE, trade is NON_COLLAPSING).

Over here ET is engine time.
Lets assume engine had a huge delay and hasn't processed any data submitted above yet.
Without any synchronization, bid/ask would completely conflate, and trade would unroll over multiple engine cycles

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET</th>
<th>ET+1</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.00</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td>100.02</td>
<td>100.03</td>
</tr>
</tbody>
</table>

However, since market data adapters will group bid/ask/trade inputs together, the engine won't let bid/ask events advance ahead of trade events since trade is NON_COLLAPSING.
NON_COLLAPSING inputs will essentially act as a barrier, not allowing events ahead of the barrier tick before the barrier is complete.
Lets assume again that the engine had a huge delay and hasn't processed any data submitted above.
With proper barrier synchronizations the engine cycles would look like this under the same conditions:

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET</th>
<th>ET+1</th>
<th>ET+2</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.01</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.03</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td>100.02</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>

Note how the last ask tick of 100.00 got held up to a separate cycle (ET+2) so that trade could tick with the correct view of bid/ask at the time of the second trade (ET+1)

As another example, lets say the engine got delayed briefly at wire time T, so it was able to process T+1 data.
Similarly it got briefly delayed at time T+4 until after T+6. The engine would be able to process all data at time T+1, T+2, T+3 and T+6, leading to this sequence of engine cycles.
The equivalent "wire time" is denoted in parenthesis

<table>
<tbody>
<tr>
<th>Event</th>
<th>ET (T+1)</th>
<th>ET+1 (T+2)</th>
<th>ET+2 (T+3)</th>
<th>ET+3 (T+5)</th>
<th>ET+4 (T+6)</th>
</tr>
&#10;<tr>
<td><strong>BID</strong></td>
<td>100.01</td>
<td><br />
</td>
<td>99.97</td>
<td>99.99</td>
<td><br />
</td>
</tr>
<tr>
<td><strong>ASK</strong></td>
<td>100.02</td>
<td>100.03</td>
<td><br />
</td>
<td><br />
</td>
<td>100.00</td>
</tr>
<tr>
<td><strong>TRADE</strong></td>
<td><br />
</td>
<td>100.02</td>
<td><br />
</td>
<td>100.03</td>
<td><br />
</td>
</tr>
</tbody>
</table>


---

# Performance Analysis with Profiling


Now that we've looked at the different execution modes, let's measure how fast and reliable they really are in each mode with our profiling tool.


The `csp.profiler` library allows users to time cycle/node executions during a graph run. There are two available utilities.

One can use these metrics to identify bottlenecks/inefficiencies in their graphs.


## Profiling a real-time `csp.graph`

The `csp.profiler` library provides a GUI for profiling real-time CSP graphs.
One can access this GUI by adding a `http_port` argument to their profiler call.

```python
with profiler.Profiler(http_port=8888) as p:
    results = csp.run(graph, starttime=st, endtime=et) # run the graph normally
```

This will open up the GUI on `localhost:8888` (as http_port=8888) which will display real-time node timing, cycle timing and memory snapshots.
Profiling stats will be calculated whenever you refresh the page or call a GET request.
Additionally, you can add the `format=json`argument (`localhost:8888?format=json`) to your request to receive the ProfilerInfo as a `JSON` object rather than the `HTML` display.

Users can add the `display_graphs=True` flag to include bar/pie charts of node execution times in the web UI.
The matplotlib package is required to use the flag.

```python
with profiler.Profiler(http_port=8888, display_graphs=True) as p:
    ...
```

<img width="466" alt="new_profiler" src="https://github.com/Point72/csp/assets/3105306/6ef692d2-16c3-4adb-ad46-a72e1017aa79">

## Saving raw profiling data to a file

Users can save individual node execution times and individual cycle execution times to a `.csv` file if they desire.
This is useful if you want to apply your own analysis e.g. calculate percentiles.
To do this, simply add the flags `node_file=<filename.csv>` or `cycle_file=<filename.csv>`

```python
with profiler.Profiler(cycle_file="cycle_data.csv", node_file="node_data.csv") as p:
    ...
```

After the graph is run, the file `node_data.csv` contains:

```
Node Type,Execution Time
count,1.9814e-05
cast_int_to_float,1.2791e-05
_time_window_updates,4.759e-06
...
```

After the graph is run, the file `cycle_data.csv` contains:

```
Execution Time
9.4757e-05
4.5205e-05
2.2873e-05
...
```

## graph_info: build-time information

Users can also extract build-time information about the graph without running it by calling `profiler.graph_info`.

The code snippet below shows how to call `graph_info`.

```python
from csp import profiler

info = profiler.graph_info(graph)
```

