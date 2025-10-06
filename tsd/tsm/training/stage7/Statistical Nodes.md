## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Working with a single-valued time series](#working-with-a-single-valued-time-series)
- [Working with a NumPy time series](#working-with-a-numpy-time-series)
- [Working with a basket of time series](#working-with-a-basket-of-time-series)
- [Cross-sectional statistics](#cross-sectional-statistics)
- [Expanding window statistics](#expanding-window-statistics)
- [Common user options](#common-user-options)
  - [Intervals](#intervals)
  - [Triggers, samplers and resets](#triggers-samplers-and-resets)
  - [Data validity](#data-validity)
  - [NaN handling](#nan-handling)
  - [Weighted statistics](#weighted-statistics)
- [Numerical stability](#numerical-stability)
  - [The `recalc` parameter](#the-recalc-parameter)

## Introduction

The `csp.stats` library provides rolling window calculations on time series data in CSP.
The goal of the library is to provide a uniform, robust interface for statistical calculations in CSP.
Each computation is a `csp.graph` which consists of one or more nodes that perform a given computation.
Users can treat these graphs as a "black box" with specified inputs and outputs as provided in the API reference.
Example statistics graphs for *mean* and *standard deviation* are provided below to give a rough idea of how the graphs work.

**Mean using a tick-specified interval**
![437686747](https://github.com/Point72/csp/assets/3105306/5586a355-e405-45c3-aa6d-c64754fd6c26)

**Standard deviation using a tick-specified interval**
![437686748](https://github.com/Point72/csp/assets/3105306/8ae2ab7a-413d-4175-89d5-5b252401a83e)

Rolling windows can either be specified by the number of ticks in the window or the time duration of the window.
Users can specify minimum window sizes for results as well as the minimum number of data points for a valid computation.
Standard NaN handling is provided with two different options.
Weighting is available for relevant stats functions such as sums, mean, covariance, and skew.

## Working with a single-valued time series

Time series of float and int types can be used for all stats functions, except those listed as "NumPy Specific".
Internally, all values are cast to float-type.
`NaN` values in the series (if applicable) are allowed and will be handled as specified by the `ignore_na` flag.

If you are performing the same calculation on many different series, **it is highly recommended that you use a NumPy array.**
NumPy array inputs result in a much smaller CSP graph which can drastically improve performance.
If different series tick asynchronously, then sometimes using single-input calculations cannot be avoided.
However, you can consider sampling your data at regularly specified intervals, and then using the sampled values to create a NumPy array which is provided to the calculation.

## Working with a NumPy time series

All statistics functions work on both single-input time series and time series of NumPy arrays.
NumPy arrays provide the ability to perform the same calculation on many different elements within the same `csp.node`, and therefore drastically reduce the overall size of the CSP graph.
The performance benefits of using NumPy arrays for large-scale computations (i.e. thousands of symbols) is order of magnitudes faster, per benchmarking.
To convert a list of individual series into a NumPy array, use the `csp.stats.list_to_numpy` conversion node.
To convert back to a basket of series, use the `csp.stats.numpy_to_list` converter.

All calculations on NumPy arrays are performed element-wise, with exception of `cov_matrix` and `corr_matrix` which are defined in the statistical sense.
Arrays of arbitrary dimension are supported, as well as array views such as transposes and slices.
The data type of arrays must be of float-type, not an int.
If your data is integer valued, convert the array to a float-type using the `astype` function in the NumPy library.
Basic mathematical operations (such as addition, multiplication etc.) are defined on NumPy array time series using NumPy's built-in functions, which allow for proper broadcasting rules.

## Working with a basket of time series

There are two ways that users can run stats function on a listbasket of time series.
If the data in the time series ticks together (or *relatively* together) then users can convert their listbasket data into a NumPy array time series
using the `list_to_numpy` node, run the calculations they want, and then convert back to a listbasket using the `numpy_to_list` node.
Since NumPy arrays only require one node per computation, whereas a list of `N` time series will require `N` nodes, this method is highly efficient even for small graphs.
Below is a diagram of the workflow for a listbasket with 2 elements.

**A sum over a listbasket with 2 elements**
![437687654](https://github.com/Point72/csp/assets/3105306/0e12b9ff-9461-497c-895d-3b1c33669235)

If the data does not tick (or is sampled) at the same time or the computations are fundamentally different in nature (i.e. different intervals), then the NumPy method will not provide the desired functionality.
Instead, if users wish to store all their individual time series in a listbasket, then they must use single input stats with standard CSP listbasket syntax.
This method is significantly slower than using NumPy arrays, since the graphs must be much larger.
However, depending on your use case, this may be unavoidable.
If possible, it is highly recommended that you consider transformations to your data that allow it to be stored in NumPy arrays, such as sampling at given intervals.

## Cross-sectional statistics

The `stats` library also exposes an option to compute cross-sectional statistics.
Cross-sectional statistics are statistics which are computed using every value in the window at each iteration.
These computations are less efficient than rolling window functions that employ smart updating.
However, some computations may have to be applied cross-sectionally, and some users may want to apply cross-sectional statistics for small window calculations that require high numerical stability.

To use cross-sectional statistics, use the `csp.stats.cross_sectional` utility to receive all data in the current window.
Then, use `csp.apply` to use your own function on the cross-sectional data.
The `cross_sectional` function allows for the same user options as standard stats functions (such as triggering and sampling).
An example of using `csp.stats.cross_sectional` is shown below:

```python
# Starttime: 2020-01-01 00:00:00
tsla_prices = {
    '2024-04-01': 180.0,
    '2024-04-02': 183.5,
    '2024-04-03': 182.0,
    '2024-04-04': 185.0,
    '2024-04-05': 190.0,
}
cs = cross_sectional(tsla_prices, interval=3, min_window=2)
cs
```

```python
{
    '2024-04-02': [180.0, 183.5]
    '2024-04-03': [180.0, 183.5, 182.0],
    '2024-04-04': [183.5, 182.0, 185.0],
    '2024-04-05': [182.0, 185.0, 190.0],
}
```

```python
# Calculate a cross-sectional mean
cs_momentum = csp.apply(cs, lambda v: v[-2] - v[0], float)
cs_momentum
```

```python
cs_momentum = {
    '2024-04-03': 182.0 - 180.0 = 2.0,
    '2024-04-04': 185.0 - 183.5 = 1.5,
    '2024-04-05': 190.0 - 182.0 = 8.0,
}
```

## Expanding window statistics

An expanding window holds all ticks of its underlying time series - in other words, the window grows unbounded as you receive more data points.
To use an expanding window, either don't specify an interval or set `interval=None`.
An example of an expanding window sum is shown below:

```python
# Starttime: 2024-04-01 00:00:00
volume = {
    '2024-04-01': 1000000,
    '2024-04-02': 1200000,
    '2024-04-03': 1100000,
    '2024-04-04': 1500000,
    '2024-04-05': 1300000,
}

sum(volume)
```

```python
{
    '2024-04-01': 1000000,
    '2024-04-02': 2200000,  # 1000000 + 1200000
    '2024-04-03': 3300000,  # 2200000 + 1100000
    '2024-04-04': 4800000,  # 3300000 + 1500000
    '2024-04-05': 6100000,  # 4800000 + 1300000
}
```

## Common user options

### Intervals

Intervals can be specified as a tick window or a time window.
Tick windows are int arguments while time windows are timedelta arguments.
For example,

- `csp.stats.mean(x, interval=4)` will calculate a rolling mean over the last 4 ticks of data.
- `csp.stats.mean(x, interval=timedelta(seconds=4))` will calculate a rolling mean over the last 4 seconds of data

Time intervals are inclusive at the right endpoint but **exclusive** at the left endpoint.
For example, if `x` ticks every one second with a value of `1`, and I call `csp.stats.sum(x, timedelta(seconds=1))`then my output will be `1` at all times.
It will not be `2`, since the left endpoint value (which ticked *exactly* one second ago) is not included.

Tick intervals include `NaN` values.
For example, a tick interval of size `10` with `9` `NaN` values in the interval will only use the single non-nan value for computations.
For more information on `NaN` handling, see the "NaN handling" section.

If no interval is specified, then the calculation will be treated as an expanding window statistic and all data will be cumulative (see the above section on Expanding Window Statistics).

### Triggers, samplers and resets

**Triggers** are optional arguments which *trigger* a computation of the statistic.
If no trigger is provided as an argument, the statistic will be computed every time `volume` ticks i.e. `volume` becomes the trigger.

```python
volume = {'2020-01-01': 1, '2020-01-02': 2, '2020-01-03': 3}
trigger = {'2020-01-02': True}

sum(volume, interval=2)
```

```python
{'2024-04-02': 2200000, '2024-04-03': 2300000}
```

```python
sum(volume, interval=2, trigger=trigger)
```

```python
# No result at 2024-04-03
{'2024-04-02': 2200000}
```

**Samplers** are optional arguments which *sample* the data.
Samplers are used to signify when the data, `volume`, *should* tick.
If no sampler is provided, the data is sampled whenever `volume` ticks i.e. `volume` becomes the sampler.

- If the sampler ticks and `volume` does as well, then the tick is treated as valid data
- If the sampler ticks but `volume` does not, then the tick is treated as `NaN` data
- If the sampler does not tick but `volume` does, then the tick is ignored completely

```python
volume = {'2024-04-01': 1000000, '2024-04-02': 1200000, '2024-04-03': 1100000}
sampler = {'2024-04-01': True, '2024-04-03': True}

sum(volume, interval=2)
```

```python
{'2024-04-02': 2200000, '2024-04-03': 2300000}
```

```python
sum(volume, interval=2, sampler=sampler)
```

```python
# Tick on 2024-04-02 is ignored
{'2024-04-03': 2100000}
```

**Resets** are optional arguments which *reset* the interval, clearing all existing data.
Whenever reset ticks, the data is cleared.
If no reset is provided, then the data is never reset.

```python
volume = {'2024-04-01': 1000000, '2024-04-02': 1200000, '2024-04-03': 1100000}
reset = {'2024-04-02 12:00:00': True}

sum(volume, interval=2)
```

```python
{'2024-04-02': 2200000, '2024-04-03': 2300000}
```

```python
sum(volume, interval=2, reset=reset)
```

```python
# Data is reset after 2024-04-02
{'2024-04-02': 2200000, '2024-04-03': 1100000}
```

**Important:** the order of operations between all three actions is as follows: reset, sample, trigger.
If all three series were to tick at the same time: the data is first reset, then sampled, and then a computation is triggered.

```python
volume = {'2024-04-01': 1000000, '2024-04-02': 1200000, '2024-04-03': 1100000}
reset = {'2024-04-03': True}

# Trigger = sampler = volume. Reset, trigger and sampler therefore all tick at 2024-04-03

sum(volume, interval=2, reset=reset)
```

```python
# the data is first reset, then 1100000 is sampled, and then the sum is computed
{'2024-04-02': 2200000, '2024-04-03': 1100000}
```

### Data validity

**Minimum window size** (`min_window`) is the smallest allowable window before returning a computation.
If a time window interval is used, then `min_window` must also be a `timedelta`.
If a tick interval is used, then `min_window` must also be an `int`.
Minimum window is a startup condition: once the minimum window size is reached, it will never go away.
For example, if you have a minimum window of 5 ticks with a 10 tick interval, once 5 ticks of data have occurred computations will always be returned when triggered.
By *default*, the minimum window size is equal to the interval itself.

```python
volume = {'2024-04-01': 1000000, '2024-04-02': 1200000, '2024-04-03': 1100000}
sum(volume, interval=2, min_window=1)
```

```python
{
    '2024-04-01': 1000000,
    '2024-04-02': 2200000,
    '2024-04-03': 2300000
}
```

```python
sum(volume, interval=timedelta(days=2), min_window=timedelta(days=1))
```

```python
# Assuming graph start time is 2024-04-01
{
    '2024-04-02': 2200000,
    '2024-04-03': 2300000
}
```

**Minimum data points** (`min_data_points`) is the number of *valid* (non-nan) data points that must exist in the current window for a valid computation.
By default, min_data_points is 0.
However, in most applications, if you are dealing with frequently NaN data you may want to ensure that stats computations provide meaningful results.
Thus, if the interval has fewer than min_data_points values, the computation is too noisy and thus NaN is returned instead.

```python
volume = {
    '2024-04-01': 1000000,
    '2024-04-02': nan,
    '2024-04-03': 1100000
}

sum(volume, interval=2)
```

```python
{
    '2024-04-02': 1000000,
    '2024-04-03': 1100000
}

sum(x, interval=2, min_data_points=2)
```

```python
# We only have 1 valid data point
{
    '2024-04-02': nan,
    '2024-04-03': nan
}
```

### NaN handling

The stats library provides a uniform interface for NaN handling.
Functions have an `ignore_na` parameter which is a bool argument (default value is `True`).

- If `ignore_na=True`, then NaN values are "ignored" in the computation but still included in the interval
- If `ignore_na=False`, then NaN values make the whole computation NaN ("poison" the interval) as long as they are present in the interval

```python
volume = {
    '2024-04-01': 1000000,
    '2024-04-02': nan,
    '2024-04-03': 1100000,
    '2024-04-04': 1200000
}

sum(volume, interval=2, ignore_na=True)
```

```python
{
    '2024-04-02': 1000000,
    '2024-04-03': 1100000,
    '2024-04-04': 2300000
}
```

```python
sum(volume, interval=2, ignore_na=False)
```

```python
# NaN at 2024-04-02 poisons the window until it rolls out
{
    '2024-04-02': nan,
    '2024-04-03': nan,
    '2024-04-04': 2300000
}
```

For exponential moving calculations, **EMA NaN handling** is slightly different.
If `ignore_na=True`, then NaN values are completely discarded.
If `ignore_na=False`, then NaN values do not poison the interval, but rather count as a tick with no data.
This affects the reweighting of past data points when the next tick with valid data is added.
For a detailed explanation, see the EMA section.

### Weighted statistics

**Weights** is an optional time-series which gives a relative weight to each data point.
Weighted statistics are available for: *sum(), mean(), var(), cov(), stddev(), sem(), corr(), skew(), kurt(), cov_matrix()* and *corr_matrix()*.
Since weights are relative, they do not need to be normalized by the user.
Weights also do not need to tick at the same time as the data, necessarily: the weights are *sampled* whenever the data sampler ticks.
For higher-order statistics such as variance, covariance, correlation, standard deviation, standard error, skewness and kurtosis, weights are interpreted as *frequency weights*.
This means that a weight of 1 corresponds to that observation occurring once and a weight of 2 signifies that observation occurring twice.

If either the data *or* its corresponding weight is NaN, then the weighted data point is collectively treated as NaN.

```python
# Daily closing prices for TSLA (used to compute Weighted Moving Average)

price = {
    '2024-04-01': 180.0,
    '2024-04-02': 182.0,
    '2024-04-03': 185.0,
    '2024-04-04': 184.0
}

# Weights favoring more recent prices
weights = {
    '2024-04-01': 1,
    '2024-04-02': 2,
    '2024-04-04': 3
}

sum(price, interval=2, weights=weights)
```

```python
# Weight of 2 applied to price=185.0, as it is sampled
{
    '2024-04-02': 542.0,  # (180 * 1 + 182 * 2)
    '2024-04-03': 549.0,  # (182 * 2 + 185 * 1)
    '2024-04-04': 738.0   # (185 * 1 + 184 * 3)
}
```

```python
mean(price, interval=2, weights=weights)
```

```python
# Weighted moving average
{
    '2024-04-02': 180.6667,  # 542 / (1 + 2)
    '2024-04-03': 183.0,     # 549 / (2 + 1)
    '2024-04-04': 184.5      # 738 / (1 + 3)
}
```

If the time-series is of type `float`, then the weights series is also of type `float`.
If the time-series is of type `np.ndarray`, then the weights series is sometimes of type `np.ndarray` and sometimes of type `float`.
For element-wise statistics *sum(), mean(), var(), stddev(), sem(), skew(), kurt()* the weights are element-wise as well.
For *cov_matrix()* and *corr_matrix(),* the weights are of type float since they apply to the data vector collectively.
Consult the individual function references for more details.

```python
# NumPy applied element-wise
# Example with multiple stocks (price vector)

price = {
    '2024-04-01': [180.0, 1000.0],
    '2024-04-02': [182.0, 990.0],
    '2024-04-03': [185.0, 995.0]
}

# Weights for each stock
weights = {
    '2024-04-01': [1,2],
    '2024-04-02': [2,1],
    '2024-04-03': [1,1]
}

sum(price, interval=2, weights=weights)

```

```python
{
    '2024-04-02': [544.0, 2990.0],
    '2024-04-03': [552.0, 2985.0]
}
```

```python
mean(price, interval=2, weights=weights)
```

```python
# Weighted moving average
{
    '2024-04-02': [181.3333, 996.6667],
    '2024-04-03': [184.0, 995.0]
}
```

## Numerical stability

Stats functions are not guaranteed to be numerically stable due to the nature of a rolling window calculation.
These functions implement online algorithms which have increased risk of floating point precision errors, especially when the data is ill-conditioned.
**Users are recommended to apply their own data cleaning** before calling these functions.
Data cleaning may include clipping large, erroneous values to be NaN or normalizing data based on historical ranges.
Cleaning can be implemented using the `csp.apply` node (see baselib documentation) with your cleaning pipeline expressed within a callable object (function).
If numerical stability is paramount, then cross-sectional calculations can be used at the cost of efficiency (see the section below on Cross-Sectional Statistics).

Where possible, `csp.stats` algorithms are chosen to maximize stability while maintaining their online efficiency.
For example, rolling variance is calculated using Welford's online algorithm and rolling sums are calculated using Kahan's algorithm if `precise=True` is set.
Floating-point error can still accumulate when the functions are used on large data streams, especially if the interval used is small in comparison to the quantity of data.
Each stats method that is prone to floating-point error exposes a **recalc parameter** which is an optional time-series argument to trigger a clean recalculation of the statistic.
The recalculation clears any accumulated floating-point error up to that point.

### The `recalc` parameter

The `recalc` parameter is an optional time-series argument designed to stop unbounded floating-point error accumulation in rolling `csp.stats` functions.
When `recalc` ticks, the next calculation of the desired statistic will be computed with all data in the window.
This clears any accumulated error from prior intervals.
The parameter is meant to be used heuristically for use cases involving large data streams and small interval sizes, causing values to be continuously added and removed from the window.
Periodically triggering a recalculation will limit the floating-point error accumulation caused by these updates; for example, a user could set `recalc` to tick every 100 intervals of their data.
The cost of triggering a recalculation is efficiency: since all data in the window must be processed, it is not as fast as doing the calculation in the standard online fashion.

A basic example using the `recalc` parameter is provided below.

```python
volume = {
    '2024-04-01': 1000000.1,
    '2024-04-02': 2000000.2,
    '2024-04-03': -1000000.1,
    '2024-04-04': -2000000.2
}
sum(volume, interval=2)
```

```python
# Floating-point error has caused the sum to not perfectly go to zero
{
    '2024-04-02': 3000000.3,
    '2024-04-03': 1000000.1000000001,
    '2024-04-04': -0.0000001
}
```

```python
recalc = {'2024-04-04': True}
sum(volume, interval=2, recalc=recalc)
```

```python
# At 2024-04-04, a clean recalculation clears the floating-point error from the previous data
{
    '2024-04-02': 3000000.3,
    '2024-04-03': 1000000.1000000001,
    '2024-04-04': 0
}
```
