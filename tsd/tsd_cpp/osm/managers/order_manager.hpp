#pragma once

#include "core/types/trade.hpp"
#include "core/types/exchange.hpp"
#include <string>
#include <future>
#include <unordered_map>

namespace tsd {
namespace osm {

// Base order manager interface
class OrderManager {
public:
    virtual ~OrderManager() = default;

    // Core order operations
    virtual std::future<std::string> send_order(const core::ExternalOrder& order) = 0;
    virtual std::future<std::string> cancel_order(const std::string& order_id) = 0;
    virtual std::future<std::string> get_order_status(const std::string& order_id) = 0;

    // Account operations
    virtual std::future<std::string> get_account_info() = 0;
    virtual std::future<std::string> get_balances() = 0;

    // Configuration
    virtual void set_credentials(const std::string& api_key, const std::string& secret) = 0;
    virtual void set_testnet(bool testnet) = 0;

    // Statistics
    struct Stats {
        size_t total_orders_sent = 0;
        size_t total_orders_filled = 0;
        size_t total_orders_cancelled = 0;
        std::chrono::milliseconds avg_response_time{0};
    };

    virtual const Stats& get_stats() const = 0;

protected:
    Stats stats_;
};

// Exchange-specific managers
class BinanceOrderManager : public OrderManager {
public:
    BinanceOrderManager(const std::string& api_key, const std::string& secret);

    std::future<std::string> send_order(const core::ExternalOrder& order) override;
    std::future<std::string> cancel_order(const std::string& order_id) override;
    std::future<std::string> get_order_status(const std::string& order_id) override;
    std::future<std::string> get_account_info() override;
    std::future<std::string> get_balances() override;

    void set_credentials(const std::string& api_key, const std::string& secret) override;
    void set_testnet(bool testnet) override;

    const Stats& get_stats() const override { return stats_; }

private:
    std::string api_key_;
    std::string secret_;
    std::string base_url_;
    bool testnet_ = false;

    std::string create_signature(const std::string& query_string);
    std::future<std::string> make_request(const std::string& method, const std::string& endpoint,
                                         const std::unordered_map<std::string, std::string>& params = {});
};

class CoinbaseOrderManager : public OrderManager {
public:
    CoinbaseOrderManager(const std::string& api_key, const std::string& secret, const std::string& passphrase);

    std::future<std::string> send_order(const core::ExternalOrder& order) override;
    std::future<std::string> cancel_order(const std::string& order_id) override;
    std::future<std::string> get_order_status(const std::string& order_id) override;
    std::future<std::string> get_account_info() override;
    std::future<std::string> get_balances() override;

    void set_credentials(const std::string& api_key, const std::string& secret) override;
    void set_testnet(bool testnet) override;

    const Stats& get_stats() const override { return stats_; }

private:
    std::string api_key_;
    std::string secret_;
    std::string passphrase_;
    std::string base_url_;
    bool testnet_ = false;

    std::string create_signature(const std::string& timestamp, const std::string& method,
                                const std::string& path, const std::string& body);
    std::future<std::string> make_request(const std::string& method, const std::string& endpoint,
                                         const std::string& body = "");
};

// Factory for creating order managers
class OrderManagerFactory {
public:
    static std::unique_ptr<OrderManager> create(core::Exchange exchange,
                                              const std::string& api_key,
                                              const std::string& secret,
                                              const std::string& passphrase = "");
};

} // namespace osm
} // namespace tsd