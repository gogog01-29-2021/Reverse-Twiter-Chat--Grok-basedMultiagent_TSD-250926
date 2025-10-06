#include "marketdata.hpp"
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace tsd {
namespace core {

std::string to_string(Side side) {
    switch (side) {
        case Side::BID: return "BID";
        case Side::ASK: return "ASK";
        default: return "UNKNOWN";
    }
}

Side side_from_string(const std::string& str) {
    if (str == "BID") return Side::BID;
    if (str == "ASK") return Side::ASK;
    throw std::invalid_argument("Unknown Side: " + str);
}

std::string MarketOrder::to_string() const {
    std::stringstream ss;
    ss << "MarketOrder( instr=" << instr.symbol()
       << ", exchange=" << tsd::core::to_string(exchange)
       << ", side=" << tsd::core::to_string(side)
       << ", price=" << std::fixed << std::setprecision(8) << price
       << ", qty=" << std::fixed << std::setprecision(8) << qty
       << ", time_exchange=" << std::chrono::duration_cast<std::chrono::milliseconds>(time_exchange.time_since_epoch()).count() << "ms"
       << ", time_received=" << std::chrono::duration_cast<std::chrono::milliseconds>(time_received.time_since_epoch()).count() << "ms )";
    return ss.str();
}

double OrderBook::best_bid() const {
    if (bids.empty()) return 0.0;

    auto max_bid = std::max_element(bids.begin(), bids.end(),
        [](const MarketOrder& a, const MarketOrder& b) {
            return a.price < b.price;
        });

    return max_bid->price;
}

double OrderBook::best_ask() const {
    if (asks.empty()) return 0.0;

    auto min_ask = std::min_element(asks.begin(), asks.end(),
        [](const MarketOrder& a, const MarketOrder& b) {
            return a.price < b.price;
        });

    return min_ask->price;
}

double OrderBook::spread() const {
    double bid = best_bid();
    double ask = best_ask();
    return (bid > 0 && ask > 0) ? ask - bid : 0.0;
}

double OrderBook::mid_price() const {
    double bid = best_bid();
    double ask = best_ask();
    return (bid > 0 && ask > 0) ? (bid + ask) / 2.0 : 0.0;
}

std::string OrderBook::to_string() const {
    std::stringstream ss;
    ss << "OrderBook( instr=" << instr.symbol()
       << ", bids=[";

    for (size_t i = 0; i < bids.size() && i < 3; ++i) { // Show first 3 levels
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(4) << bids[i].price << "@" << bids[i].qty;
    }

    ss << "], asks=[";

    for (size_t i = 0; i < asks.size() && i < 3; ++i) { // Show first 3 levels
        if (i > 0) ss << ", ";
        ss << std::fixed << std::setprecision(4) << asks[i].price << "@" << asks[i].qty;
    }

    ss << "], time_exchange=" << std::chrono::duration_cast<std::chrono::milliseconds>(time_exchange.time_since_epoch()).count() << "ms"
       << ", time_received=" << std::chrono::duration_cast<std::chrono::milliseconds>(time_received.time_since_epoch()).count() << "ms )";

    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Side& side) {
    return os << to_string(side);
}

std::ostream& operator<<(std::ostream& os, const TwoWayPrice& price) {
    return os << "TwoWayPrice( bid=" << std::fixed << std::setprecision(4) << price.bid_price
              << "@" << price.bid_qty << ", ask=" << price.ask_price << "@" << price.ask_qty << " )";
}

std::ostream& operator<<(std::ostream& os, const MarketOrder& order) {
    return os << order.to_string();
}

std::ostream& operator<<(std::ostream& os, const OrderBook& book) {
    return os << book.to_string();
}

} // namespace core
} // namespace tsd