#include "order_manager.hpp"
#include <iostream>
#include <sstream>
#include <thread>

namespace tsd {
namespace osm {

// BinanceOrderManager implementation
BinanceOrderManager::BinanceOrderManager(const std::string& api_key, const std::string& secret)
    : api_key_(api_key), secret_(secret) {
    base_url_ = testnet_ ? "https://testnet.binance.vision" : "https://api.binance.com";
}

std::future<std::string> BinanceOrderManager::send_order(const core::ExternalOrder& order) {
    return std::async(std::launch::async, [this, order]() -> std::string {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::unordered_map<std::string, std::string> params;
        params["symbol"] = order.instr.symbol();
        params["side"] = core::to_string(order.order_side);
        params["type"] = core::to_string(order.order_type);
        params["quantity"] = std::to_string(order.qty.value_or(0.0));
        params["price"] = std::to_string(order.price);
        params["timeInForce"] = core::to_string(order.time_in_force);
        params["newClientOrderId"] = order.client_order_id;

        // Simulate API call
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Mock response
        std::stringstream response;
        response << R"({
            "symbol": ")" << params["symbol"] << R"(",
            "orderId": 12345,
            "orderListId": -1,
            "clientOrderId": ")" << order.client_order_id << R"(",
            "transactTime": )" << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() << R"(,
            "price": ")" << order.price << R"(",
            "origQty": ")" << order.qty.value_or(0.0) << R"(",
            "executedQty": "0.00000000",
            "cummulativeQuoteQty": "0.00000000",
            "status": "NEW",
            "timeInForce": ")" << core::to_string(order.time_in_force) << R"(",
            "type": ")" << core::to_string(order.order_type) << R"(",
            "side": ")" << core::to_string(order.order_side) << R"(",
            "exchange": "BINANCE"
        })";

        auto end_time = std::chrono::high_resolution_clock::now();
        auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        stats_.total_orders_sent++;
        stats_.avg_response_time = (stats_.avg_response_time + response_time) / 2;

        return response.str();
    });
}

std::future<std::string> BinanceOrderManager::cancel_order(const std::string& order_id) {
    return std::async(std::launch::async, [this, order_id]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        stats_.total_orders_cancelled++;
        return R"({"orderId": )" + order_id + R"(, "status": "CANCELLED"})";
    });
}

std::future<std::string> BinanceOrderManager::get_order_status(const std::string& order_id) {
    return std::async(std::launch::async, [order_id]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return R"({"orderId": )" + order_id + R"(, "status": "FILLED"})";
    });
}

std::future<std::string> BinanceOrderManager::get_account_info() {
    return std::async(std::launch::async, []() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        return R"({"makerCommission": 10, "takerCommission": 10, "buyerCommission": 0})";
    });
}

std::future<std::string> BinanceOrderManager::get_balances() {
    return std::async(std::launch::async, []() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
        return R"({"balances": [{"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"}]})";
    });
}

void BinanceOrderManager::set_credentials(const std::string& api_key, const std::string& secret) {
    api_key_ = api_key;
    secret_ = secret;
}

void BinanceOrderManager::set_testnet(bool testnet) {
    testnet_ = testnet;
    base_url_ = testnet_ ? "https://testnet.binance.vision" : "https://api.binance.com";
}

std::string BinanceOrderManager::create_signature(const std::string& query_string) {
    // Mock signature for demonstration
    return "mock_signature_" + std::to_string(std::hash<std::string>{}(query_string + secret_));
}

std::future<std::string> BinanceOrderManager::make_request(const std::string& method, const std::string& endpoint,
                                                          const std::unordered_map<std::string, std::string>& params) {
    return std::async(std::launch::async, [method, endpoint, params]() -> std::string {
        // Mock HTTP request
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return R"({"result": "success"})";
    });
}

// CoinbaseOrderManager implementation
CoinbaseOrderManager::CoinbaseOrderManager(const std::string& api_key, const std::string& secret, const std::string& passphrase)
    : api_key_(api_key), secret_(secret), passphrase_(passphrase) {
    base_url_ = testnet_ ? "https://api-public.sandbox.exchange.coinbase.com" : "https://api.exchange.coinbase.com";
}

std::future<std::string> CoinbaseOrderManager::send_order(const core::ExternalOrder& order) {
    return std::async(std::launch::async, [this, order]() -> std::string {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::this_thread::sleep_for(std::chrono::milliseconds(60));

        std::stringstream response;
        response << R"({
            "id": "12345-abcdef",
            "product_id": ")" << order.instr.symbol() << R"(",
            "side": ")" << (order.order_side == core::OrderSide::BUY ? "buy" : "sell") << R"(",
            "type": ")" << (order.order_type == core::OrderType::LIMIT ? "limit" : "market") << R"(",
            "price": ")" << order.price << R"(",
            "size": ")" << order.qty.value_or(0.0) << R"(",
            "time_in_force": ")" << core::to_string(order.time_in_force) << R"(",
            "status": "pending",
            "exchange": "COINBASE"
        })";

        auto end_time = std::chrono::high_resolution_clock::now();
        auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        stats_.total_orders_sent++;
        stats_.avg_response_time = (stats_.avg_response_time + response_time) / 2;

        return response.str();
    });
}

std::future<std::string> CoinbaseOrderManager::cancel_order(const std::string& order_id) {
    return std::async(std::launch::async, [this, order_id]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        stats_.total_orders_cancelled++;
        return R"([{"id": ")" + order_id + R"(", "status": "cancelled"}])";
    });
}

std::future<std::string> CoinbaseOrderManager::get_order_status(const std::string& order_id) {
    return std::async(std::launch::async, [order_id]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        return R"({"id": ")" + order_id + R"(", "status": "done"})";
    });
}

std::future<std::string> CoinbaseOrderManager::get_account_info() {
    return std::async(std::launch::async, []() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(45));
        return R"([{"id": "account1", "currency": "USD", "balance": "10000.00"}])";
    });
}

std::future<std::string> CoinbaseOrderManager::get_balances() {
    return std::async(std::launch::async, []() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        return R"([{"currency": "BTC", "balance": "1.00000000", "available": "1.00000000"}])";
    });
}

void CoinbaseOrderManager::set_credentials(const std::string& api_key, const std::string& secret) {
    api_key_ = api_key;
    secret_ = secret;
}

void CoinbaseOrderManager::set_testnet(bool testnet) {
    testnet_ = testnet;
    base_url_ = testnet_ ? "https://api-public.sandbox.exchange.coinbase.com" : "https://api.exchange.coinbase.com";
}

std::string CoinbaseOrderManager::create_signature(const std::string& timestamp, const std::string& method,
                                                  const std::string& path, const std::string& body) {
    // Mock signature for demonstration
    return "mock_cb_signature_" + std::to_string(std::hash<std::string>{}(timestamp + method + path + body + secret_));
}

std::future<std::string> CoinbaseOrderManager::make_request(const std::string& method, const std::string& endpoint,
                                                           const std::string& body) {
    return std::async(std::launch::async, [method, endpoint, body]() -> std::string {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        return R"({"result": "success"})";
    });
}

// OrderManagerFactory implementation
std::unique_ptr<OrderManager> OrderManagerFactory::create(core::Exchange exchange,
                                                         const std::string& api_key,
                                                         const std::string& secret,
                                                         const std::string& passphrase) {
    switch (exchange) {
        case core::Exchange::BINANCE:
            return std::make_unique<BinanceOrderManager>(api_key, secret);
        case core::Exchange::COINBASE:
            return std::make_unique<CoinbaseOrderManager>(api_key, secret, passphrase);
        default:
            throw std::runtime_error("Unsupported exchange: " + core::to_string(exchange));
    }
}

} // namespace osm
} // namespace tsd