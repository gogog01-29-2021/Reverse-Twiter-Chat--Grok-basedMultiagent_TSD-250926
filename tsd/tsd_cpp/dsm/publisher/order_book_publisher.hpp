#pragma once

#include "core/types/marketdata.hpp"
#include "core/types/instrument.hpp"
#include <functional>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <random>

namespace tsd {
namespace dsm {

class OrderBookPublisher {
public:
    using Callback = std::function<void(const core::OrderBook&)>;

    OrderBookPublisher(const std::string& endpoint = "tcp://localhost:5555");
    ~OrderBookPublisher();

    // Publishing methods
    void publish(const core::OrderBook& book);
    void publish_async(const core::OrderBook& book);

    // Subscription methods
    void subscribe(Callback callback);
    void unsubscribe();

    // Control methods
    void start();
    void stop();

    // Configuration
    void set_endpoint(const std::string& endpoint);
    void set_publish_interval(std::chrono::milliseconds interval);

    // Statistics
    struct Stats {
        size_t total_published = 0;
        size_t total_subscribers = 0;
        std::chrono::milliseconds avg_latency{0};
    };

    const Stats& get_stats() const { return stats_; }

private:
    std::string endpoint_;
    std::vector<Callback> callbacks_;

    std::atomic<bool> running_{false};
    std::atomic<bool> publishing_{false};

    std::thread publisher_thread_;
    std::mutex callbacks_mutex_;
    std::mutex publish_mutex_;

    std::chrono::milliseconds publish_interval_{1};
    Stats stats_;

    void publisher_loop();
    void notify_subscribers(const core::OrderBook& book);
};

} // namespace dsm
} // namespace tsd