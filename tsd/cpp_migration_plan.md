# C/C++ Migration Plan for CSP Trading System

## 1. Core Architecture Design

### Data Structures (Replace csp.Struct)
```cpp
// core/types/base.hpp
template<typename T>
struct TimeSeries {
    std::chrono::nanoseconds timestamp;
    T value;
};

// core/types/trade.hpp
enum class TimeInForce { IOC, GTC, DAY };
enum class OrderSide { BUY, SELL };
enum class OrderType { LIMIT, MARKET };

struct ExternalOrder {
    Instrument instr;
    Exchange exchange;
    OrderSide order_side;
    OrderType order_type;
    double price;
    std::optional<double> qty;
    std::optional<double> quote_order_qty;
    TimeInForce time_in_force;
    int64_t time_sent;
    std::string portfolio;
    std::string client_order_id;
};

struct Trade {
    Instrument instr;
    double price;
    double qty;
    OrderSide side;
};
```

### Graph Execution Engine (Replace @csp.graph/@csp.node)
```cpp
// core/engine/node.hpp
template<typename Input, typename Output>
class Node {
public:
    virtual Output process(const TimeSeries<Input>& input) = 0;
    virtual void setup() {}
    virtual void teardown() {}
};

// core/engine/graph.hpp
class Graph {
private:
    std::vector<std::unique_ptr<BaseNode>> nodes;
    std::vector<Edge> edges;

public:
    template<typename NodeType>
    void add_node(std::unique_ptr<NodeType> node);

    void connect(NodeId from, NodeId to);
    void run(std::chrono::nanoseconds start_time);
};

// core/engine/runtime.hpp
class CSPRuntime {
public:
    void run_graph(Graph& graph, std::chrono::nanoseconds start_time);

private:
    EventScheduler scheduler;
    MemoryPool memory_pool;
};
```

## 2. Key Components to Implement

### Time-Series Operations
```cpp
// Replace csp.const(), csp.curve()
template<typename T>
class ConstNode : public Node<void, T> {
    T value;
public:
    ConstNode(T val) : value(val) {}
    T process(const TimeSeries<void>&) override { return value; }
};

template<typename T>
class CurveNode : public Node<void, T> {
    std::vector<std::pair<std::chrono::nanoseconds, T>> data;
    size_t current_index = 0;
public:
    CurveNode(std::vector<std::pair<std::chrono::nanoseconds, T>> curve_data);
    T process(const TimeSeries<void>&) override;
};
```

### Accumulation Operations
```cpp
// Replace csp.accum()
template<typename T>
class AccumNode : public Node<T, T> {
    T accumulated_value{};
public:
    T process(const TimeSeries<T>& input) override {
        accumulated_value += input.value;
        return accumulated_value;
    }
};
```

### Market Data Streaming
```cpp
// dsm/core/publisher.hpp
class OrderBookPublisher {
public:
    void publish(const OrderBook& book);
    void subscribe(std::function<void(const OrderBook&)> callback);
};

// dsm/core/subscriber.hpp
class OrderBookSubscriber {
public:
    void connect(const std::string& endpoint);
    TimeSeries<OrderBook> receive();
};
```

## 3. Build System (CMake)

### Project Structure
```
tsd_cpp/
├── CMakeLists.txt
├── core/
│   ├── types/
│   │   ├── trade.hpp
│   │   ├── marketdata.hpp
│   │   └── instrument.hpp
│   ├── engine/
│   │   ├── node.hpp
│   │   ├── graph.hpp
│   │   └── runtime.hpp
│   └── utils/
├── dsm/
│   ├── publisher/
│   └── subscriber/
├── osm/
│   ├── managers/
│   └── utils/
├── examples/
│   ├── basic_example.cpp
│   └── order_flow_test.cpp
└── tests/
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.20)
project(TSD_CPP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Core library
add_library(tsd_core
    core/engine/runtime.cpp
    core/engine/graph.cpp
    core/types/trade.cpp
    core/types/marketdata.cpp
)

target_include_directories(tsd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tsd_core ${ZMQ_LIBRARIES} Threads::Threads)

# Examples
add_executable(basic_example examples/basic_example.cpp)
target_link_libraries(basic_example tsd_core)

# Tests
enable_testing()
add_subdirectory(tests)
```

## 4. Migration Strategy

1. **Phase 1**: Core data structures and basic nodes
2. **Phase 2**: Graph execution engine and runtime
3. **Phase 3**: Market data streaming (DSM equivalent)
4. **Phase 4**: Order management (OSM equivalent)
5. **Phase 5**: Strategy framework (TSM equivalent)

## 5. Performance Optimizations

- Zero-copy message passing
- Lock-free queues for inter-node communication
- Memory pools for frequent allocations
- SIMD operations for numerical computations
- Template metaprogramming for compile-time optimizations

This design provides the same CSP functionality as Python but with C++ performance and no external dependencies.