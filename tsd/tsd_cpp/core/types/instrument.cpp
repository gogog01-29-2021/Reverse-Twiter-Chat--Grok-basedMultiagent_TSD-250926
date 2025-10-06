#include "instrument.hpp"
#include <stdexcept>

namespace tsd {
namespace core {

std::string to_string(Currency currency) {
    switch (currency) {
        case Currency::BTC: return "BTC";
        case Currency::ETH: return "ETH";
        case Currency::USDT: return "USDT";
        case Currency::USDC: return "USDC";
        case Currency::USD: return "USD";
        case Currency::EUR: return "EUR";
        case Currency::GBP: return "GBP";
        case Currency::JPY: return "JPY";
        case Currency::BNB: return "BNB";
        case Currency::ADA: return "ADA";
        case Currency::DOT: return "DOT";
        case Currency::LINK: return "LINK";
        case Currency::SOL: return "SOL";
        case Currency::MATIC: return "MATIC";
        case Currency::AVAX: return "AVAX";
        case Currency::UNKNOWN: return "UNKNOWN";
        default: return "UNKNOWN";
    }
}

Currency currency_from_string(const std::string& str) {
    if (str == "BTC") return Currency::BTC;
    if (str == "ETH") return Currency::ETH;
    if (str == "USDT") return Currency::USDT;
    if (str == "USDC") return Currency::USDC;
    if (str == "USD") return Currency::USD;
    if (str == "EUR") return Currency::EUR;
    if (str == "GBP") return Currency::GBP;
    if (str == "JPY") return Currency::JPY;
    if (str == "BNB") return Currency::BNB;
    if (str == "ADA") return Currency::ADA;
    if (str == "DOT") return Currency::DOT;
    if (str == "LINK") return Currency::LINK;
    if (str == "SOL") return Currency::SOL;
    if (str == "MATIC") return Currency::MATIC;
    if (str == "AVAX") return Currency::AVAX;
    return Currency::UNKNOWN;
}

std::ostream& operator<<(std::ostream& os, const Currency& currency) {
    return os << to_string(currency);
}

std::ostream& operator<<(std::ostream& os, const Spot& spot) {
    return os << "Spot(" << spot.base << "/" << spot.term << ")";
}

std::ostream& operator<<(std::ostream& os, const Future& future) {
    return os << "Future(" << future.base << "/" << future.term << " " << future.expiry << ")";
}

std::ostream& operator<<(std::ostream& os, const Instrument& instrument) {
    switch (instrument.type) {
        case InstrumentType::SPOT:
            return os << instrument.spot;
        case InstrumentType::FUTURE:
            return os << instrument.future;
        default:
            return os << "Unknown Instrument";
    }
}

} // namespace core
} // namespace tsd