#include "core/engine/graph.hpp"
#include "core/engine/runtime.hpp"
#include "core/engine/node.hpp"
#include "core/types/trade.hpp"
#include "core/types/instrument.hpp"
#include <iostream>
#include <vector>

using namespace tsd::core;

// Trade PnL calculation node
class TradePnLNode : public Node<Trade, double> {
private:
    double running_pnl_ = 0.0;
    double position_ = 0.0;
    double avg_price_ = 0.0;

public:
    TradePnLNode(NodeId id, const std::string& name = "trade_pnl")
        : Node<Trade, double>(id, name) {}

    double compute(const TimeSeries<Trade>& input) override {
        const auto& trade = input.value;

        double trade_qty = trade.qty;
        if (trade.side == OrderSide::SELL) {
            trade_qty = -trade_qty;
        }

        // Calculate PnL for this trade
        double trade_pnl = 0.0;
        if (position_ != 0.0) {
            // If we have existing position, calculate unrealized PnL
            trade_pnl = position_ * (trade.price - avg_price_);
        }

        // Update position and average price
        double new_position = position_ + trade_qty;
        if (new_position == 0.0) {
            // Closing position
            running_pnl_ += trade_pnl;
            avg_price_ = 0.0;
        } else if ((position_ > 0 && trade_qty > 0) || (position_ < 0 && trade_qty < 0)) {
            // Adding to position
            avg_price_ = (avg_price_ * position_ + trade.price * trade_qty) / new_position;
        } else {
            // Reducing position
            double realized_pnl = (trade.price - avg_price_) * (-trade_qty);
            running_pnl_ += realized_pnl;
        }

        position_ = new_position;

        // Current unrealized PnL
        double unrealized_pnl = position_ * (trade.price - avg_price_);
        double total_pnl = running_pnl_ + unrealized_pnl;

        std::cout << "Trade: " << (trade.side == OrderSide::BUY ? "BUY" : "SELL")
                  << " " << trade.qty << " @ " << trade.price
                  << " | Position: " << position_
                  << " | Avg Price: " << avg_price_
                  << " | Total PnL: " << total_pnl << std::endl;

        return total_pnl;
    }
};

// Trade curve node with realistic trade data
class TradeCurveNode : public BaseNode {
private:
    std::vector<TimeSeries<Trade>> trades_;
    size_t current_index_ = 0;

public:
    TradeCurveNode(NodeId id, const std::string& name = "trade_curve")
        : BaseNode(id, name) {
        register_output("output");

        auto start_time = std::chrono::high_resolution_clock::now();
        Spot btc_usd{Currency::BTC, Currency::USD};

        trades_.push_back({start_time, Trade{btc_usd, 50000.0, 0.1, OrderSide::BUY}});
        trades_.push_back({start_time + std::chrono::milliseconds(100), Trade{btc_usd, 50100.0, 0.05, OrderSide::BUY}});
        trades_.push_back({start_time + std::chrono::milliseconds(200), Trade{btc_usd, 50200.0, 0.08, OrderSide::SELL}});
        trades_.push_back({start_time + std::chrono::milliseconds(300), Trade{btc_usd, 50150.0, 0.07, OrderSide::SELL}});
        trades_.push_back({start_time + std::chrono::milliseconds(400), Trade{btc_usd, 50300.0, 0.1, OrderSide::BUY}});

        std::sort(trades_.begin(), trades_.end());
    }

    void process(TimePoint current_time) override {
        if (current_index_ < trades_.size()) {
            const auto trade = trades_[current_index_++].value;
            set_output("output", TimeSeries<Trade>{current_time, trade});
        } else {
            std::any empty;
            set_output("output", empty);
        }
    }
};

int main() {
    std::cout << "=== TSM Stage 2 - Trade PnL Example ===" << std::endl;

    // Build graph
    GraphBuilder builder;

    auto trades_id = builder.add_node<TradeCurveNode>();
    auto pnl_id = builder.add_node<TradePnLNode>();
    auto pnl_print_id = builder.add_node<PrintNode<double>>("PnL: $", "pnl_print");

    // Connect nodes
    builder.connect(trades_id, pnl_id)
           .connect(pnl_id, pnl_print_id);

    auto graph = builder.build();

    // Execute
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::milliseconds(500);

    std::cout << "Running trade PnL calculation..." << std::endl;

    try {
        run(*graph, start_time, end_time);
        std::cout << "Trade PnL calculation completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Trade PnL Example Complete ===" << std::endl;
    return 0;
}