#include "core/engine/graph.hpp"
#include "core/engine/runtime.hpp"
#include "core/engine/node.hpp"
#include "core/types/trade.hpp"
#include "core/types/instrument.hpp"
#include <iostream>
#include <vector>

using namespace tsd::core;

// Custom curve node for trades (equivalent to csp.curve)
class TradesCurveNode : public Node<void, Trade> {
private:
    std::vector<TimeSeries<Trade>> trades_;
    size_t current_index_ = 0;

public:
    TradesCurveNode(NodeId id, std::vector<TimeSeries<Trade>> trades, const std::string& name = "trades_curve")
        : Node<void, Trade>(id, name), trades_(std::move(trades)) {
        // Sort trades by timestamp
        std::sort(trades_.begin(), trades_.end());
    }

    Trade compute(const TimeSeries<void>&) override {
        if (current_index_ < trades_.size()) {
            return trades_[current_index_++].value;
        }
        // Return empty trade when done
        return Trade{};
    }
};

// Accumulation node for trade quantities
class TradeSizeAccumNode : public Node<Trade, int> {
private:
    int cumulative_qty_ = 0;

public:
    TradeSizeAccumNode(NodeId id, const std::string& name = "cumqty")
        : Node<Trade, int>(id, name) {}

    int compute(const TimeSeries<Trade>& input) override {
        cumulative_qty_ += static_cast<int>(input.value.qty);
        return cumulative_qty_;
    }
};

// Trade printing node
class TradePrintNode : public Node<Trade, Trade> {
private:
    std::string prefix_;

public:
    TradePrintNode(NodeId id, const std::string& prefix = "trades:", const std::string& name = "trade_print")
        : Node<Trade, Trade>(id, name), prefix_(prefix) {}

    Trade compute(const TimeSeries<Trade>& input) override {
        const auto& trade = input.value;
        std::cout << prefix_ << "Trade( price=" << trade.price
                  << ", size=" << static_cast<int>(trade.qty) << " )" << std::endl;
        return trade;
    }
};

int main() {
    std::cout << "=== TSD C++ Trading Example ===" << std::endl;

    // Create trading data (equivalent to Python example)
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<TimeSeries<Trade>> trade_data = {
        {start_time, Trade{Spot{Currency::BTC, Currency::USD}, 100.01, 200, OrderSide::BUY}},
        {start_time + std::chrono::seconds(1), Trade{Spot{Currency::BTC, Currency::USD}, 100.02, 300, OrderSide::BUY}}
    };

    // Build graph
    GraphBuilder builder;

    auto trades_id = builder.add_node<TradesCurveNode>(std::move(trade_data));
    auto sizes_id = builder.add_node<TradeSizeAccumNode>();
    auto trade_print_id = builder.add_node<TradePrintNode>();
    auto cumqty_print_id = builder.add_node<PrintNode<int>>("cumqty:", "cumqty_print");

    // Connect nodes
    builder.connect(trades_id, trade_print_id)
           .connect(trades_id, sizes_id)
           .connect(sizes_id, cumqty_print_id);

    auto graph = builder.build();

    // Execute
    std::cout << "Running trading graph..." << std::endl;

    auto end_time = start_time + std::chrono::seconds(2);

    try {
        run(*graph, start_time, end_time);
        std::cout << "Trading graph completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error running trading graph: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Trading Example Complete ===" << std::endl;
    return 0;
}