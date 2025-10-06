#include "exchange.hpp"

namespace tsd {
namespace core {

std::string to_string(Exchange exchange) {
    switch (exchange) {
        case Exchange::BINANCE: return "BINANCE";
        case Exchange::COINBASE: return "COINBASE";
        case Exchange::BYBIT: return "BYBIT";
        case Exchange::OKX: return "OKX";
        case Exchange::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

Exchange exchange_from_string(const std::string& str) {
    if (str == "BINANCE") return Exchange::BINANCE;
    if (str == "COINBASE") return Exchange::COINBASE;
    if (str == "BYBIT") return Exchange::BYBIT;
    if (str == "OKX") return Exchange::OKX;
    return Exchange::UNKNOWN;
}

std::ostream& operator<<(std::ostream& os, const Exchange& exchange) {
    return os << to_string(exchange);
}

} // namespace core
} // namespace tsd