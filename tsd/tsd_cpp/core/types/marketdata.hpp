#pragma once

#include "base.hpp"
#include "instrument.hpp"
#include "exchange.hpp"
#include <vector>

namespace tsd {
namespace core {

enum class Side {
    BID,
    ASK
};

std::string to_string(Side side);
Side side_from_string(const std::string& str);

struct TwoWayPrice {
    double bid_price;
    double ask_price;
    double bid_qty;
    double ask_qty;
    int64_t time_exchange;   // nanoseconds since epoch
    int64_t time_received;   // nanoseconds since epoch

    TwoWayPrice() = default;
    TwoWayPrice(double bid_price, double ask_price, double bid_qty, double ask_qty,
               int64_t time_exchange = 0, int64_t time_received = 0)
        : bid_price(bid_price), ask_price(ask_price), bid_qty(bid_qty), ask_qty(ask_qty),
          time_exchange(time_exchange), time_received(time_received) {}
};

struct MarketOrder {
    Instrument instr;
    Exchange exchange;
    Side side;
    double price;
    double qty;
    TimePoint time_exchange;
    TimePoint time_received;

    MarketOrder() = default;
    MarketOrder(Instrument instr, Exchange exchange, Side side, double price, double qty,
               TimePoint time_exchange = TimePoint{}, TimePoint time_received = TimePoint{})
        : instr(instr), exchange(exchange), side(side), price(price), qty(qty),
          time_exchange(time_exchange == TimePoint{} ? std::chrono::high_resolution_clock::now() : time_exchange),
          time_received(time_received == TimePoint{} ? std::chrono::high_resolution_clock::now() : time_received) {}

    std::string to_string() const;
};

struct OrderBook {
    Instrument instr;
    std::vector<MarketOrder> bids;
    std::vector<MarketOrder> asks;
    TimePoint time_exchange;
    TimePoint time_received;

    OrderBook() = default;
    OrderBook(Instrument instr, std::vector<MarketOrder> bids, std::vector<MarketOrder> asks,
             TimePoint time_exchange = TimePoint{}, TimePoint time_received = TimePoint{})
        : instr(instr), bids(std::move(bids)), asks(std::move(asks)),
          time_exchange(time_exchange == TimePoint{} ? std::chrono::high_resolution_clock::now() : time_exchange),
          time_received(time_received == TimePoint{} ? std::chrono::high_resolution_clock::now() : time_received) {}

    // Helper methods
    double best_bid() const;
    double best_ask() const;
    double spread() const;
    double mid_price() const;

    std::string to_string() const;
};

// Stream operators
std::ostream& operator<<(std::ostream& os, const Side& side);
std::ostream& operator<<(std::ostream& os, const TwoWayPrice& price);
std::ostream& operator<<(std::ostream& os, const MarketOrder& order);
std::ostream& operator<<(std::ostream& os, const OrderBook& book);

} // namespace core
} // namespace tsd