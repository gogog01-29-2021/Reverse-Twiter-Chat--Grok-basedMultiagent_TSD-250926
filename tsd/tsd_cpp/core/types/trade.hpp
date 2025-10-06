#pragma once

#include "base.hpp"
#include "instrument.hpp"
#include "exchange.hpp"
#include <string>
#include <optional>

namespace tsd {
namespace core {

enum class TimeInForce {
    IOC,  // Immediate or Cancel
    GTC,  // Good Till Cancelled
    DAY   // Good for Day
};

enum class OrderSide {
    BUY,
    SELL
};

enum class OrderType {
    LIMIT,
    MARKET
};

// Convert enums to strings for serialization
std::string to_string(TimeInForce tif);
std::string to_string(OrderSide side);
std::string to_string(OrderType type);

TimeInForce time_in_force_from_string(const std::string& str);
OrderSide order_side_from_string(const std::string& str);
OrderType order_type_from_string(const std::string& str);

struct ExternalOrder {
    Instrument instr;
    Exchange exchange;
    OrderSide order_side;
    OrderType order_type;

    double price;
    std::optional<double> qty;                // base currency quantity
    std::optional<double> quote_order_qty;    // quote currency quantity

    TimeInForce time_in_force;
    int64_t time_sent;                        // nanoseconds since epoch
    std::string portfolio;
    std::string client_order_id;

    ExternalOrder() = default;

    // Constructor with automatic client_order_id generation
    ExternalOrder(Instrument instr, Exchange exchange, OrderSide side, OrderType type,
                  double price, std::optional<double> qty, std::optional<double> quote_qty,
                  TimeInForce tif, int64_t time_sent, const std::string& portfolio);

private:
    std::string generate_client_order_id(const std::string& portfolio);
};

struct Trade {
    Instrument instr;
    double price;
    double qty;
    OrderSide side;
    TimePoint timestamp;

    Trade() = default;
    Trade(Instrument instr, double price, double qty, OrderSide side, TimePoint ts = TimePoint{})
        : instr(instr), price(price), qty(qty), side(side),
          timestamp(ts == TimePoint{} ? std::chrono::high_resolution_clock::now() : ts) {}
};

struct ExecutionReport {
    std::string order_id;
    std::string client_order_id;
    Instrument instr;
    Exchange exchange;
    OrderSide side;
    OrderType type;

    double price;
    double qty;
    double filled_qty;
    double remaining_qty;

    std::string status;  // NEW, PARTIALLY_FILLED, FILLED, CANCELLED, etc.
    TimePoint timestamp;

    ExecutionReport() = default;
};

} // namespace core
} // namespace tsd