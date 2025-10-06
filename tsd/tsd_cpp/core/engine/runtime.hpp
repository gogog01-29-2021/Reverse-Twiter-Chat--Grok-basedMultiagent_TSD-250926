#pragma once

#include "graph.hpp"
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace tsd {
namespace core {

// Event-driven scheduler for CSP execution
class EventScheduler {
public:
    struct Event {
        TimePoint timestamp;
        NodeId node_id;
        std::string input_name;
        std::any data;

        bool operator>(const Event& other) const {
            return timestamp > other.timestamp; // For min-heap
        }
    };

    void schedule_event(TimePoint when, NodeId node, const std::string& input, const std::any& data);
    bool has_events() const;
    Event get_next_event();
    void clear();

private:
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;
    mutable std::mutex queue_mutex_;
};

// Memory pool for efficient allocation of time-series data
class MemoryPool {
public:
    MemoryPool(size_t block_size = 4096, size_t initial_blocks = 10);
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);

    // Statistics
    size_t allocated_bytes() const { return allocated_bytes_.load(); }
    size_t total_allocations() const { return total_allocations_.load(); }

private:
    struct Block {
        void* data;
        size_t size;
        size_t used;
        Block* next;
    };

    Block* current_block_;
    const size_t block_size_;
    std::atomic<size_t> allocated_bytes_{0};
    std::atomic<size_t> total_allocations_{0};
    mutable std::mutex pool_mutex_;

    void allocate_new_block();
};

// Main CSP runtime engine
class CSPRuntime {
public:
    CSPRuntime();
    ~CSPRuntime();

    // Graph execution modes
    void run_graph(Graph& graph, TimePoint start_time, TimePoint end_time);
    void run_graph_realtime(Graph& graph, Duration step_interval = std::chrono::milliseconds(1));

    // Runtime control
    void stop();
    void pause();
    void resume();

    // Configuration
    void set_max_threads(size_t count) { max_threads_ = count; }
    void set_memory_pool_size(size_t bytes) { memory_pool_size_ = bytes; }

    // Statistics
    struct Stats {
        size_t total_events_processed = 0;
        size_t total_nodes_executed = 0;
        Duration total_execution_time{0};
        size_t memory_allocated = 0;
    };

    const Stats& get_stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    EventScheduler scheduler_;
    MemoryPool memory_pool_;

    size_t max_threads_;
    size_t memory_pool_size_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex runtime_mutex_;
    std::condition_variable runtime_cv_;

    Stats stats_;

    // Execution helpers
    void execute_graph_step(Graph& graph, TimePoint current_time);
    void process_node(BaseNode* node, TimePoint current_time);
};

// Convenience function for simple graph execution
void run(Graph& graph, TimePoint start_time, TimePoint end_time = TimePoint{});

} // namespace core
} // namespace tsd