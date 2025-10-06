#include "core/engine/graph.hpp"
#include "core/engine/runtime.hpp"
#include "core/engine/node.hpp"
#include "core/types/trade.hpp"
#include "core/types/marketdata.hpp"
#include "dsm/publisher/order_book_publisher.hpp"
#include "osm/managers/order_manager.hpp"
#include <iostream>
#include <vector>

using namespace tsd::core;
using namespace tsd::dsm;
using namespace tsd::osm;

// Market data feed node
class MarketDataFeedNode : public Node<void, OrderBook> {
private:
    std::vector<TimeSeries<OrderBook>> market_data_;
    size_t current_index_ = 0;

public:
    MarketDataFeedNode(NodeId id, const std::string& name = "market_feed")
        : Node<void, OrderBook>(id, name) {

        auto start_time = std::chrono::high_resolution_clock::now();
        Spot btc_usd{Currency::BTC, Currency::USD};

        // Create realistic market data
        for (int i = 0; i < 10; ++i) {
            std::vector<MarketOrder> bids, asks;

            // Generate bid levels
            for (int j = 0; j < 5; ++j) {
                double price = 50000.0 - j * 10.0 - i * 5.0;
                double qty = 1.0 + j * 0.5;
                bids.emplace_back(btc_usd, Exchange::BINANCE, Side::BID, price, qty);
            }

            // Generate ask levels
            for (int j = 0; j < 5; ++j) {
                double price = 50010.0 + j * 10.0 + i * 5.0;
                double qty = 1.0 + j * 0.5;
                asks.emplace_back(btc_usd, Exchange::BINANCE, Side::ASK, price, qty);
            }

            OrderBook book{btc_usd, bids, asks};
            auto timestamp = start_time + std::chrono::milliseconds(i * 100);
            market_data_.emplace_back(timestamp, book);
        }
    }

    OrderBook compute(const TimeSeries<void>&) override {
        if (current_index_ < market_data_.size()) {
            return market_data_[current_index_++].value;
        }
        return OrderBook{}; // Empty when done
    }
};

// Signal generation node
class TradingSignalNode : public Node<OrderBook, std::string> {
private:
    double last_mid_price_ = 0.0;
    int signal_count_ = 0;

public:
    TradingSignalNode(NodeId id, const std::string& name = "signal_generator")
        : Node<OrderBook, std::string>(id, name) {}

    std::string compute(const TimeSeries<OrderBook>& input) override {
        const auto& book = input.value;
        double current_mid = book.mid_price();

        std::string signal = "HOLD";

        if (last_mid_price_ > 0) {
            double price_change = (current_mid - last_mid_price_) / last_mid_price_;

            if (price_change > 0.001) { // 0.1% increase
                signal = "BUY";
            } else if (price_change < -0.001) { // 0.1% decrease
                signal = "SELL";
            }
        }

        last_mid_price_ = current_mid;
        signal_count_++;

        std::cout << "Signal #" << signal_count_
                  << " | Mid Price: $" << current_mid
                  << " | Signal: " << signal << std::endl;

        return signal;
    }
};

// Order execution node
class OrderExecutionNode : public Node<std::string, std::string> {
private:
    std::unique_ptr<OrderManager> order_manager_;
    Spot instrument_;

public:
    OrderExecutionNode(NodeId id, const std::string& name = "order_executor")
        : Node<std::string, std::string>(id, name),
          instrument_{Currency::BTC, Currency::USD} {

        // Create order manager with test credentials
        order_manager_ = OrderManagerFactory::create(
            Exchange::BINANCE,
            "test_api_key",
            "test_secret"
        );
        order_manager_->set_testnet(true);
    }

    std::string compute(const TimeSeries<std::string>& input) override {
        const auto& signal = input.value;

        if (signal == "HOLD") {
            return "NO_ORDER";
        }

        // Create order based on signal
        OrderSide side = (signal == "BUY") ? OrderSide::BUY : OrderSide::SELL;
        double price = (signal == "BUY") ? 50005.0 : 49995.0; // Simple pricing
        double qty = 0.01; // Small test quantity

        ExternalOrder order{
            instrument_,
            Exchange::BINANCE,
            side,
            OrderType::LIMIT,
            price,
            qty,
            std::nullopt, // quote_order_qty
            TimeInForce::GTC,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count(),
            "test_portfolio"
        };

        // Send order asynchronously
        auto future_response = order_manager_->send_order(order);

        try {
            auto response = future_response.get();
            std::cout << "Order executed: " << signal
                      << " | Response: " << response.substr(0, 100) << "..." << std::endl;
            return "ORDER_SENT";
        } catch (const std::exception& e) {
            std::cerr << "Order execution failed: " << e.what() << std::endl;
            return "ORDER_FAILED";
        }
    }
};

int main() {
    std::cout << "=== Complete Trading System Example ===" << std::endl;

    // Build comprehensive trading graph
    GraphBuilder builder;

    // Market data pipeline
    auto market_feed_id = builder.add_node<MarketDataFeedNode>();
    auto book_print_id = builder.add_node<PrintNode<OrderBook>>("Market Data: ", "book_printer");

    // Signal generation
    auto signal_id = builder.add_node<TradingSignalNode>();
    auto signal_print_id = builder.add_node<PrintNode<std::string>>("Trading Signal: ", "signal_printer");

    // Order execution
    auto execution_id = builder.add_node<OrderExecutionNode>();
    auto execution_print_id = builder.add_node<PrintNode<std::string>>("Execution Status: ", "execution_printer");

    // Connect the pipeline
    builder.connect(market_feed_id, book_print_id)
           .connect(market_feed_id, signal_id)
           .connect(signal_id, signal_print_id)
           .connect(signal_id, execution_id)
           .connect(execution_id, execution_print_id);

    auto graph = builder.build();

    // Execute the complete trading system
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(2);

    std::cout << "Starting complete trading system..." << std::endl;

    try {
        run(*graph, start_time, end_time);
        std::cout << "Trading system completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Trading system error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Complete Trading System Example Complete ===" << std::endl;
    return 0;
}