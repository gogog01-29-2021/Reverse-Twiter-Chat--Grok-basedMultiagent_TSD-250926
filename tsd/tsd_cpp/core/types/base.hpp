#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace tsd {
namespace core {

// Time type for high-precision timestamps
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Duration = std::chrono::nanoseconds;

// Base time-series value container
template<typename T>
struct TimeSeries {
    TimePoint timestamp;
    T value;

    TimeSeries() = default;
    TimeSeries(TimePoint ts, T val) : timestamp(ts), value(std::move(val)) {}

    // Comparison operators for sorting by time
    bool operator<(const TimeSeries<T>& other) const {
        return timestamp < other.timestamp;
    }

    bool operator==(const TimeSeries<T>& other) const {
        return timestamp == other.timestamp;
    }
};

// Forward declarations
class BaseNode;
class Graph;

// Node identifier
using NodeId = uint64_t;

// Edge represents connection between nodes
struct Edge {
    NodeId from;
    NodeId to;
    std::string from_output;
    std::string to_input;
};

// Specialization for void value type
template<>
struct TimeSeries<void> {
    TimePoint timestamp;

    TimeSeries() = default;
    explicit TimeSeries(TimePoint ts) : timestamp(ts) {}

    bool operator<(const TimeSeries<void>& other) const {
        return timestamp < other.timestamp;
    }

    bool operator==(const TimeSeries<void>& other) const {
        return timestamp == other.timestamp;
    }
};

} // namespace core
} // namespace tsd
