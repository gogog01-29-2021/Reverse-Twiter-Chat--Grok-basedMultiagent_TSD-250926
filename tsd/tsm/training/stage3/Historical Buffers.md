## Table of Contents

- [csp.Struct](#cspstruct)
  - [`csp.Struct` definition](#cspstruct-definition)
  - [Special handling at graph time](#special-handling-at-graph-time)
  - [List fields](#list-fields)
  - [Available methods](#available-methods)
  - [Note on inheritance](#note-on-inheritance)
- [State Handling of Composite Structures: Cart Example](#state-handling-of-composite-structures-cart-example)
  - [Structured data with `csp.Struct`](#structured-data-with-cspstruct)
  - [Track cart updates](#track-cart-updates)
  - [Create workflow graph](#create-workflow-graph)
  - [Execute the graph](#execute-the-graph)
- [Use historical values](#use-historical-values)
  - [Historical Buffers](#historical-buffers)
  - [Historical Range Access](#historical-range-access)

# csp.Struct

`csp.Struct` is a native, first-class CSP type that should be used for struct-like data ( key, values ). `csp.Struct` is implemented as a high performant C++ object and CSP C++ adapters are able to create / access them efficiently from C++.

In general its recommended to use `csp.Struct` for any object-like data required on your timeseries rather than plain old python object unless there is good reason not to.

## `csp.Struct` definition

`csp.Struct` types need to declare all their fields as annotated fields, for example:

```python
class MyData(csp.Struct):
    a: int
    b: str = 'default'
    c: list

    def some_method(self):
        return a * 2

>>> MyData()
MyData( a='<unset>', b='default', c='<unset>' )
```

The variables `a`, `b`, `c` here define the struct members (similar to `__slots__` on python objects, structs can not have any other attributes set on them other than the ones listed here).

**Defaults**: Note that members can be defined with default values, as `b` is here.

**Unset fields**: Note that stgruct fields can be "unset". Fields can be checked for existence with `hasattr(o, field)` and can be removed with `del o.field`.

**Methods**: Note that you can define methods on structs just like any other python object.

## Special handling at graph time Q.Graph Time?

While building your graph, if you have an edge that represents a `csp.Struct` type you can access a member of that struct at graph time. What this means is that when you do `edge.field` in your graph code, you will get a new edge that will tick with the value of that field. CSP will implicitly inject a `csp.node` that will extract that field whenever the struct timeseries ticks. Note that if the struct ticks and the field is unset, the field's edge will not tick. Here's an example of this in practice:

```python
import csp
from datetime import datetime

class Trade(csp.Struct):
    price: float
    size: int

@csp.graph
def my_graph():
    trades = csp.curve(Trade,
                       [(datetime(2020, 1, 1), Trade(price=100.01, size=200)),
                       (datetime(2020, 1, 1, 0, 0, 1), Trade(price=100.01, size=300))]
             )

    sizes = trades.size
    cumqty = csp.accum(sizes)

    csp.print('trades', trades)
    csp.print('cumqty', cumqty)


csp.run( my_graph, starttime = datetime( 2020, 1, 1 ))


>>> 2020-01-01 00:00:00 trades:Trade( price=100.01, size=200 )
>>> 2020-01-01 00:00:00 cumqty:200
>>> 2020-01-01 00:00:01 trades:Trade( price=100.01, size=300 )
>>> 2020-01-01 00:00:01 cumqty:500
```

`trades` is defined as a timeseries of `Trade` objects. On line 13 we access the `size` field of the `trades` timeseries, then accumulate the sizes to get `cumqty` edge.

## List fields

List fields in a `csp.Struct` can be specified as three different types - untyped Python list, typed regular list and typed `FastList`.

A Python list field keeps its value as a Python object, it is not typed. It can be created by annotating the field as `list`.

A regular list field is typed and internally implemented as a subclass of the Python list class, so it behaves exactly like Python list and passes the `isinstance` check for list. It can be created by annotating the field as `typing.List[T]` or `[T]`.

A `FastList` field is typed and a more efficient implementation, implementing all Python list operations natively in C++. However, it doesn't pass `isinstance` check for list. It can be created by annotating the field as `csp.impl.types.typing_utils.FastList[T]`.

Example of using Python list field:

```python
import csp

class A(csp.Struct):
    a: list

s = A(a = [1, 'x'])

s.a.append(True)

print(f"Using Python list field: value {s.a}, type {type(s.a)}, is Python list: {isinstance(s.a, list)}")


>>> Using Python list field: value [1, 'x', True], type <class 'list'>, is Python list: True
```

Example of using regular list field:

```python
from typing import List
import csp

class A(csp.Struct):
    a: List[int]

s = A(a = [1, 2])

s.a.append(3)

print(f"Using list field: value {s.a}, type {type(s.a)}, is Python list: {isinstance(s.a, list)}")


>>> Using list field: value [1, 2, 3], type <class '_cspimpl.PyStructList'>, is Python list: True
```

Example of using `FastList` field:

```python
import csp
from csp.impl.types.typing_utils import FastList

class A(csp.Struct):
    a: FastList[int]

s = A(a = [1, 2])

s.a.append(3)

print(f"Using FastList field: value {s.a}, type {type(s.a)}, is Python list: {isinstance(s.a, list)}")


>>> Using FastList field: value [1, 2, 3], type <class '_cspimpl.PyStructFastList'>, is Python list: False
```

## Available methods

- **`clear(self)`** clear all fields on the struct
- **`collectts(self, **kwargs)`**: `kwargs` expects key/values of struct fields and time series to populate the struct. This will return an Edge representing a ticking struct created from all the ticking inputs provided. Structs will only be generated from inputs that actively ticked in the given engine cycle (see `fromts` to create struct from all valid inputs)
- **`copy(self)`**: return a shallow copy of the struct
- **`copy_from(self, rhs)`**: copy data from `rhs` into this instance. `rhs` must be of the same type or a derived type. Copy will include unsetting fields unset in the `rhs`.
- **`update_from(self, rhs)`**: copy only the set fields from the `rhs` into this instance
- **`update(self, **kwargs)`**: in this instance, set the provided fields with the provided values
- **`fromts(self, trigger=None, /, **kwargs)`**: similar to `collectts` above, `fromts` will create a ticking Struct timeseries from the valid values of all the provided inputs whenever any of them tick. `trigger` is an optional position-only argument which is used as a trigger timeseries for when to convert inputs into a struct tick. By default any input tick will generate a new struct of valid inputs
- **`from_dict(self, dict)`**: convert a regular python dict to an instance of the struct
- **`metadata(self)`**: returns the struct's metadata as a dictionary of key : type pairs
- **`to_dict(self, callback=None, preserve_enums=False)`**: convert struct instance to a python dictionary, if callback is not None it is invoked for any values encountered when processing the struct that are not basic Python types, datetime types, tuples, lists, sets, dicts, csp.Structs, or csp.Enums. If preserve_enums=True, then enums are not converted to strings in the output dictionary.
- **`postprocess_to_dict(cls, obj)`**: if this method is defined on a `csp.Struct` class, the `to_dict` method will invoke this method on the dict obtained after processing the struct of this class to allow for further customization.
- **`to_dict_depr(self)`**: convert struct instance to a python dictionary. \[DEPRECATED\]: This is a python only and slower implementation of to_dict which will be removed.
- **`to_json(self, callback=lambda x: x)`**: convert struct instance to a json string, callback is invoked for any values encountered when processing the struct that are not basic Python types, datetime types, tuples, lists, dicts, csp.Structs, or csp.Enums. The callback should convert the unhandled type to a combination of the known types.
- **`all_fields_set(self)`**: returns `True` if all the fields on the struct are set. Note that this will not recursively check sub-struct fields

## Note on inheritance

`csp.Struct` types may inherit from each other, but **multiple inheritance is not supported**. Composition is usually a good choice in absence of multiple inheritance.

---

# State Handling of Composite Structures: Cart Example

Now that we have learned how to maintain and leverage past values, we will now look at how to use time-series data from composite structures in conjunction with state variables in practice.

We have looked at the features of CSP nodes and graphs, as well as how to run an application using `csp.run`. In this tutorial, we will apply what we learned in [stage1](../stage1/CSP%20Node.md) and [More with CSP](../stage2/CSP%20Graph.md) to build a basic retail app which maintains an online shopping cart.


We will also introduce two important new concepts: the [[`csp.Struct`](../stage3/Historical%20Buffers.md)] data structure and multi-output nodes using `csp.Outputs`.

Our application will track a customer's shopping cart and apply a 10% discount for any items added to the cart in the first minute. Check out the complete code [[e5_retail_cart.py](../stage2/e5_retail_cart.py)].

## Structured data with `csp.Struct`

An individual item in a shopping cart consists of many fields; for example, the product's name, quantity and cost. The shopping cart itself may contain a list of these items as a field, plus a user ID or name. We also want to store updates to the shopping cart in an organized data structure, which has fields indicating the item in question and whether it was added or removed.
 

## Track cart updates

Recall from [More with CSP](../stage2/CSP%20Graph.md) that we can store state variables in a `csp.node` using a `csp.state` block. We will create a node that tracks updates to a user's cart by storing the `Cart` struct as a state variable named `s_cart`.

> [!TIP]
> By convention, state variables are prefixed with `s_` for readability.

A CSP node can return multiple named outputs. To annotate a multi-output node, we use `csp.Outputs` syntax for the return type annotation. To tick out each named value, we use the `csp.output` function. After each update event, we will tick out the total value of the user's cart and the number of items present.

To apply a discount for all items added in the first minute, we can use an alarm. We discussed how to use a `csp.alarm` as an internal time-series in the [Poisson counter example](../stage2/CSP%20Graph.md). We will only update the cart when the user adds, removes or purchases items. We need to know what the active discount rate to apply is but we don't need to trigger an update when it changes. To achieve this, we make the alarm time-series `discount` a *passive* input.

A *passive* input is a time-series input that will not cause the node to execute when it ticks. When we access the input within the node, we always get its most recent value. The opposite of passive inputs are *active* inputs, which trigger a node to compute upon a tick. So far, every input we've worked with has been an active input. We will set the discount input to be passive at graph startup.

> [!TIP]
> By default, all `csp.ts` inputs are active. You can change the activity of an input at any point during execution by using `csp.make_passive` or `csp.make_active`.

```python
from csp import ts
from datetime import timedelta
from functools import reduce


@csp.node
def update_cart(event: ts[CartUpdate], user_id: int) -> csp.Outputs(total=ts[float], num_items=ts[int]):
  """
  Track of the cart total and number of items.
  """
  with csp.alarms():
    discount = csp.alarm(float)

  with csp.state():
    # create an empty shopping cart
    s_cart = Cart(user_id=user_id, items=[])

  with csp.start():
    csp.make_passive(discount)
    csp.schedule_alarm(discount, timedelta(), 0.9)  # 10% off for the first minute
    csp.schedule_alarm(discount, timedelta(minutes=1), 1.0)  # full price after!

  if csp.ticked(event):
    if event.add:
      # apply current discount
      event.item.cost *= discount
      s_cart.items.append(event.item)
    else:
      # remove the given qty of the item
      new_items = []
      remaining_qty = event.item.qty
      for item in s_cart.items:
        if item.symbol == event.item.symbol:
          if item.qty > remaining_qty:
            item.qty -= remaining_qty
            new_items.append(item)
          else:
            remaining_qty -= item.qty
        else:
          new_items.append(item)
      s_cart.items = new_items

  current_total = reduce(lambda a, b: a + b.cost * b.qty, s_cart.items, 0)
  current_num_items = reduce(lambda a, b: a + b.qty, s_cart.items, 0)
  csp.output(total=current_total, num_items=current_num_items)
```

## Create workflow graph

To create example cart updates, we will use a [[`csp.curve`](../stage2/CSP%20Graph.md)] like we have in previous examples. The `csp.curve` replays a list of events at specific times.

```python
st = datetime(2020, 1, 1)

@csp.graph
def my_graph():
    # Example cart updates
    events = csp.curve(
        CartUpdate,
        [
            # Add 1 unit of X at $10 plus a 10% discount
            (st + timedelta(seconds=15), CartUpdate(item=Item(name="X", cost=10, qty=1), add=True)),
            # Add 2 units of Y at $15 each, plus a 10% discount
            (st + timedelta(seconds=30), CartUpdate(item=Item(name="Y", cost=15, qty=2), add=True)),
            # Remove 1 unit of Y
            (st + timedelta(seconds=45), CartUpdate(item=Item(name="Y", qty=1), add=False)),
            # Add 1 unit of Z at $20 but no discount, since our minute expired
            (st + timedelta(seconds=75), CartUpdate(item=Item(name="Z", cost=20, qty=1), add=True)),
        ],
    )

    csp.print("Events", events)

    current_cart = update_cart(events, user_id=42)

    csp.print("Cart number of items", current_cart.num_items)
    csp.print("Cart total", current_cart.total)
```

## Execute the graph

Execute the program and observe the outputs that our shopping cart provides.

```python
def main():
    csp.run(my_graph, starttime=st)
```

```raw
2020-01-01 00:00:15 Events:CartUpdate( item=Item( name=X, cost=10.0, qty=1 ), add=True )
2020-01-01 00:00:15 Cart total:9.0
2020-01-01 00:00:15 Cart number of items:1
2020-01-01 00:00:30 Events:CartUpdate( item=Item( name=Y, cost=15.0, qty=2 ), add=True )
2020-01-01 00:00:30 Cart total:36.0
2020-01-01 00:00:30 Cart number of items:3
2020-01-01 00:00:45 Events:CartUpdate( item=Item( name=Y, cost=<unset>, qty=1 ), add=False )
2020-01-01 00:00:45 Cart total:22.5
2020-01-01 00:00:45 Cart number of items:2
2020-01-01 00:01:15 Events:CartUpdate( item=Item( name=Z, cost=20.0, qty=1 ), add=True )
2020-01-01 00:01:15 Cart total:42.5
2020-01-01 00:01:15 Cart number of items:3
```

---


# Use historical values

## Historical Buffers

CSP can provide access to historical input data as well.
By default only the last value of an input is kept in memory, however one can request history to be kept on an input either by number of ticks or by time using **csp.set_buffering_policy.**

The methods **csp.value_at**, **csp.time_at** and **csp.item_at** can be used to retrieve historical input values.
Each node should call **csp.set_buffering_policy** to make sure that its inputs are configured to store sufficiently long history for correct implementation.
For example, let's assume that we have a stream of data and we want to create equally sized buckets from the data.
A possible implementation of such a node would be:

```python
@csp.node
def data_bin_generator(bin_size: int, input: ts['T']) -> ts[['T']]:
    with csp.start():
        assert bin_size > 0
        # This makes sure that input stores at least bin_size entries
        csp.set_buffering_policy(input, tick_count=bin_size)
    if csp.ticked(input) and (csp.num_ticks(input) % bin_size == 0):
        return [csp.value_at(input, -i) for i in range(bin_size)]
```

In this example, we use **`csp.set_buffering_policy(input, tick_count=bin_size)`** to ensure that the buffer history contains at least **`bin_size`** elements.
Note that an input can be shared by multiple nodes, if multiple nodes provide size requirements, the buffer size would be resolved to the maximum size to support all requests.

Alternatively, **`csp.set_buffering_policy`** supports a **`timedelta`** parameter **`tick_history`** instead of **`tick_count`.**
If **`tick_history`** is provided, the buffer will scale dynamically to ensure that any period of length **`tick_history`** will fit into the history buffer.

To identify when there are enough samples to construct a bin we use **`csp.num_ticks(input) % bin_size == 0`**.
The function **`csp.num_ticks`** returns the number or total ticks for a given time series.
NOTE: The actual size of the history buffer is usually less than **`csp.num_ticks`** as buffer is dynamically truncated to satisfy the set policy.

The past values in this example are accessed using **`csp.value_at`**.
The various historical access methods take the same arguments and return the value, time and tuple of `(time,value)` respectively:

- **`csp.value_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns **value** of the timeseries at requested `index_or_time`
- **`csp.time_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns **datetime** of the timeseries at requested `index_or_time`
- **`csp.item_at`**`(ts, index_or_time, duplicate_policy=DuplicatePolicy.LAST_VALUE, default=UNSET)`: returns tuple of `(datetime,value)` of the timeseries at requested `index_or_time`
  - **`ts`**: the name of the input
  - **`index_or_time`**:
    - If providing an **index**, this represents how many ticks back to rereieve **and should be \<= 0**.
      0 indicates the current value, -1 is the previous value, etc.
    - If providing **time** one can either provide a datetime for absolute time, or a timedelta for how far back to access.
      **NOTE** that timedelta must be negative to represent time in the past..
  - **`duplicate_policy`**: when requesting history by datetime or timedelta, its possible that there could be multiple values that match the given time.
    **`duplicate_policy`** can be provided to control the behavior of what to return in this case.
    The default policy is to return the LAST_VALUE that exists at the given time.
  - **`default`**: value to be returned if the requested time is out of the history bounds (if default is not provided and a request is out of bounds an exception will be raised).

The following demonstrate a possible way to compute a rolling sum for the past N ticks. Please note that this is for demonstration purposes only and is not efficient. A more efficient
vectorized version can be seen below, though even that would not be recommended for a rolling sum since csp.stats.sum would be even more efficient with its C++ impl in-line calculation

```python
@csp.node
def rolling_sum(x:ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        csp.set_buffering_policy(x, tick_count=tick_count)

    if csp.ticked(x):
        return sum(csp.value_at(x, -i) for i in range(min(csp.num_ticks(x), tick_count)))
```

## Historical Range Access

In similar fashion, the methods **`csp.values_at`**, **`csp.times_at`** and **`csp.items_at`** can be used to retrieve a range of historical input values as numpy arrays.
The sample_sum example above can be accomplished more efficiently with range access:

```python
@csp.node
def rolling_sum(x:ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        csp.set_buffering_policy(x, tick_count=tick_count)

    if csp.ticked(x):
        return csp.values_at(x).sum()
```

The past values in this example are accessed using **`csp.values_at`**.
The various historical access methods take the same arguments and return the value, time and tuple of `(times,values)` respectively:

- **`csp.values_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns values in specified range as a numpy array
- **`csp.times_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns times in specified range as a numpy array
- **`csp.items_at`**`(ts, start_index_or_time, end_index_or_time, start_index_policy=TimeIndexPolicy.INCLUSIVE, end_index_policy=TimeIndexPolicy.INCLUSIVE)`:
  returns a tuple of (times, values) numpy arrays
  - **`ts`** - the name of the input
  - **`start_index_or_time`**:
    - If providing an **index**, this represents how many ticks back to retrieve **and should be \<= 0**.
      0 indicates the current value, -1 is the previous value, etc.
    - If providing **time** one can either provide a datetime for absolute time, or a timedelta for how far back to access.
      **NOTE that timedelta must be negative** to represent time in the past..
    - If **None** is provided, the range will begin "from the beginning" - i.e., the oldest tick in the buffer.
  - **`end_index_or_time`**: same as start_index_or_time
    - If **None** is provided, the range will go "until the end" - i.e., the newest tick in the buffer.
  - **`start_index_policy`**: only for use with datetime/timedelta as the start and end parameters.
    - **`TimeIndexPolicy.INCLUSIVE`**: if there is a tick exactly at the requested time, include it
    - **`TimeIndexPolicy.EXCLUSIVE`**: if there is a tick exactly at the requested time, exclude it
    - **`TimeIndexPolicy.EXTRAPOLATE`**: if there is a tick at the beginning timestamp, include it.
      Otherwise, if there is a tick before the beginning timestamp, force a tick at the beginning timestamp with the prevailing value at the time.
  - **`end_index_policy`**: only for use with datetime/timedelta and the start and end parameters.
    - **`TimeIndexPolicy.INCLUSIVE`**: if there is a tick exactly at the requested time, include it
    - **`TimeIndexPolicy.EXCLUSIVE`**: if there is a tick exactly at the requested time, exclude it
    - **`TimeIndexPolicy.EXTRAPOLATE`**: if there is a tick at the end timestamp, include it.
      Otherwise, if there is a tick before the end timestamp, force a tick at the end timestamp with the prevailing value at the time

Range access is optimized at the C++ layer and for this reason its far more efficient than calling the single value access methods in a loop, and they should be substituted in where possible.

Below is a rolling average example to illustrate the use of timedelta indexing.
Note that `timedelta(seconds=-n_seconds)` is equivalent to `csp.now() - timedelta(seconds=n_seconds)`, since datetime indexing is supported.

```python
@csp.node
def rolling_average(x: ts[float], n_seconds: int) -> ts[float]:
    with csp.start():
        assert n_seconds > 0
        csp.set_buffering_policy(x, tick_history=timedelta(seconds=n_seconds))
    if csp.ticked(x):
        avg = np.mean(csp.values_at(x, timedelta(seconds=-n_seconds), timedelta(seconds=0),
                                      csp.TimeIndexPolicy.INCLUSIVE, csp.TimeIndexPolicy.INCLUSIVE))
        csp.output(avg)
```

When accessing all elements within the buffering policy window like this, it would be more succinct to pass None as the start and end time, but datetime/timedelta allows for more general use (e.g. rolling average between 5 seconds and 1 second ago, or average specifically between 9:30:00 and 10:00:00)



