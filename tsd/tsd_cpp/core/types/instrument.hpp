#pragma once

#include <string>
#include <ostream>

namespace tsd {
namespace core {

enum class Currency {
    BTC, ETH, USDT, USDC, USD, EUR, GBP, JPY,
    BNB, ADA, DOT, LINK, SOL, MATIC, AVAX,
    UNKNOWN
};

std::string to_string(Currency currency);
Currency currency_from_string(const std::string& str);

struct Spot {
    Currency base;
    Currency term;

    Spot() : base(Currency::UNKNOWN), term(Currency::UNKNOWN) {}
    Spot(Currency base, Currency term) : base(base), term(term) {}

    std::string symbol() const {
        return to_string(base) + to_string(term);
    }

    bool operator==(const Spot& other) const {
        return base == other.base && term == other.term;
    }

    bool operator!=(const Spot& other) const {
        return !(*this == other);
    }
};

struct Future {
    Currency base;
    Currency term;
    std::string expiry;  // YYYYMMDD format

    Future() : base(Currency::UNKNOWN), term(Currency::UNKNOWN) {}
    Future(Currency base, Currency term, const std::string& expiry)
        : base(base), term(term), expiry(expiry) {}

    std::string symbol() const {
        return to_string(base) + to_string(term) + expiry;
    }

    bool operator==(const Future& other) const {
        return base == other.base && term == other.term && expiry == other.expiry;
    }

    bool operator!=(const Future& other) const {
        return !(*this == other);
    }
};

// Variant-like union for different instrument types
enum class InstrumentType {
    SPOT,
    FUTURE
};

struct Instrument {
    InstrumentType type;
    union {
        Spot spot;
        Future future;
    };

    // Constructors
    Instrument() : type(InstrumentType::SPOT), spot() {}
    Instrument(const Spot& s) : type(InstrumentType::SPOT), spot(s) {}
    Instrument(const Future& f) : type(InstrumentType::FUTURE), future(f) {}

    // Copy constructor
    Instrument(const Instrument& other) : type(other.type) {
        if (type == InstrumentType::SPOT) {
            spot = other.spot;
        } else {
            future = other.future;
        }
    }

    // Assignment operator
    Instrument& operator=(const Instrument& other) {
        if (this != &other) {
            type = other.type;
            if (type == InstrumentType::SPOT) {
                spot = other.spot;
            } else {
                future = other.future;
            }
        }
        return *this;
    }

    // Destructor
    ~Instrument() {
        if (type == InstrumentType::SPOT) {
            spot.~Spot();
        } else {
            future.~Future();
        }
    }

    std::string symbol() const {
        switch (type) {
            case InstrumentType::SPOT:
                return spot.symbol();
            case InstrumentType::FUTURE:
                return future.symbol();
            default:
                return "UNKNOWN";
        }
    }

    bool operator==(const Instrument& other) const {
        if (type != other.type) return false;
        switch (type) {
            case InstrumentType::SPOT:
                return spot == other.spot;
            case InstrumentType::FUTURE:
                return future == other.future;
            default:
                return false;
        }
    }

    bool operator!=(const Instrument& other) const {
        return !(*this == other);
    }
};

// Stream operators for debugging
std::ostream& operator<<(std::ostream& os, const Currency& currency);
std::ostream& operator<<(std::ostream& os, const Spot& spot);
std::ostream& operator<<(std::ostream& os, const Future& future);
std::ostream& operator<<(std::ostream& os, const Instrument& instrument);

} // namespace core
} // namespace tsd