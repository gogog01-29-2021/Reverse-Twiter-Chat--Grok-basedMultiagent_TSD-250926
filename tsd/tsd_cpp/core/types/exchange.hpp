#pragma once

#include <string>
#include <ostream>

namespace tsd {
namespace core {

enum class Exchange {
    BINANCE,
    COINBASE,
    BYBIT,
    OKX,
    UNKNOWN
};

std::string to_string(Exchange exchange);
Exchange exchange_from_string(const std::string& str);

std::ostream& operator<<(std::ostream& os, const Exchange& exchange);

} // namespace core
} // namespace tsd