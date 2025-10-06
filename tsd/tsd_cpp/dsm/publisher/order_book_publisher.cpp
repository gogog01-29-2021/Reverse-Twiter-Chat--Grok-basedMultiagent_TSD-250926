#include "order_book_publisher.hpp"
#include <iostream>
#include <chrono>

namespace tsd {
namespace dsm {

OrderBookPublisher::OrderBookPublisher(const std::string& endpoint)
    : endpoint_(endpoint) {
}

OrderBookPublisher::~OrderBookPublisher() {
    stop();
}

void OrderBookPublisher::publish(const core::OrderBook& book) {
    std::lock_guard<std::mutex> lock(publish_mutex_);

    auto start_time = std::chrono::high_resolution_clock::now();

    notify_subscribers(book);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    stats_.total_published++;
    stats_.avg_latency = (stats_.avg_latency + latency) / 2;
}

void OrderBookPublisher::publish_async(const core::OrderBook& book) {
    if (!publishing_) {
        publishing_ = true;
        std::thread([this, book]() {
            publish(book);
            publishing_ = false;
        }).detach();
    }
}

void OrderBookPublisher::subscribe(Callback callback) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.push_back(callback);
    stats_.total_subscribers = callbacks_.size();
}

void OrderBookPublisher::unsubscribe() {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    callbacks_.clear();
    stats_.total_subscribers = 0;
}

void OrderBookPublisher::start() {
    if (!running_) {
        running_ = true;
        publisher_thread_ = std::thread(&OrderBookPublisher::publisher_loop, this);
    }
}

void OrderBookPublisher::stop() {
    if (running_) {
        running_ = false;
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
    }
}

void OrderBookPublisher::set_endpoint(const std::string& endpoint) {
    endpoint_ = endpoint;
}

void OrderBookPublisher::set_publish_interval(std::chrono::milliseconds interval) {
    publish_interval_ = interval;
}

void OrderBookPublisher::publisher_loop() {
    while (running_) {
        std::this_thread::sleep_for(publish_interval_);
        // Publisher loop implementation would go here
        // For now, just maintain the thread
    }
}

void OrderBookPublisher::notify_subscribers(const core::OrderBook& book) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    for (const auto& callback : callbacks_) {
        try {
            callback(book);
        } catch (const std::exception& e) {
            std::cerr << "Error in subscriber callback: " << e.what() << std::endl;
        }
    }
}

} // namespace dsm
} // namespace tsd