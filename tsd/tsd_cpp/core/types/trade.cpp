#include "trade.hpp"
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>

namespace tsd {
namespace core {

std::string to_string(TimeInForce tif) {
    switch (tif) {
        case TimeInForce::IOC: return "IOC";
        case TimeInForce::GTC: return "GTC";
        case TimeInForce::DAY: return "DAY";
        default: return "UNKNOWN";
    }
}

std::string to_string(OrderSide side) {
    switch (side) {
        case OrderSide::BUY: return "BUY";
        case OrderSide::SELL: return "SELL";
        default: return "UNKNOWN";
    }
}

std::string to_string(OrderType type) {
    switch (type) {
        case OrderType::LIMIT: return "LIMIT";
        case OrderType::MARKET: return "MARKET";
        default: return "UNKNOWN";
    }
}

TimeInForce time_in_force_from_string(const std::string& str) {
    if (str == "IOC") return TimeInForce::IOC;
    if (str == "GTC") return TimeInForce::GTC;
    if (str == "DAY") return TimeInForce::DAY;
    throw std::invalid_argument("Unknown TimeInForce: " + str);
}

OrderSide order_side_from_string(const std::string& str) {
    if (str == "BUY") return OrderSide::BUY;
    if (str == "SELL") return OrderSide::SELL;
    throw std::invalid_argument("Unknown OrderSide: " + str);
}

OrderType order_type_from_string(const std::string& str) {
    if (str == "LIMIT") return OrderType::LIMIT;
    if (str == "MARKET") return OrderType::MARKET;
    throw std::invalid_argument("Unknown OrderType: " + str);
}

ExternalOrder::ExternalOrder(Instrument instr, Exchange exchange, OrderSide side, OrderType type,
                           double price, std::optional<double> qty, std::optional<double> quote_qty,
                           TimeInForce tif, int64_t time_sent, const std::string& portfolio)
    : instr(instr), exchange(exchange), order_side(side), order_type(type),
      price(price), qty(qty), quote_order_qty(quote_qty),
      time_in_force(tif), time_sent(time_sent), portfolio(portfolio),
      client_order_id(generate_client_order_id(portfolio)) {
}

std::string ExternalOrder::generate_client_order_id(const std::string& portfolio) {
    // Generate UUID-like string
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> hex_dis(0, 15);

    std::stringstream ss;
    ss << portfolio << "_";

    // Generate 8 character hex string
    for (int i = 0; i < 8; ++i) {
        ss << std::hex << hex_dis(gen);
    }

    return ss.str();
}

} // namespace core
} // namespace tsd