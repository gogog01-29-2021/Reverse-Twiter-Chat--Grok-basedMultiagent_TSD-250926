#include "runtime.hpp"
#include <iostream>
#include <thread>

namespace tsd {
namespace core {

// EventScheduler implementation
void EventScheduler::schedule_event(TimePoint when, NodeId node, const std::string& input, const std::any& data) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    event_queue_.push({when, node, input, data});
}

bool EventScheduler::has_events() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !event_queue_.empty();
}

EventScheduler::Event EventScheduler::get_next_event() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (event_queue_.empty()) {
        throw std::runtime_error("No events available");
    }
    auto event = event_queue_.top();
    event_queue_.pop();
    return event;
}

void EventScheduler::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!event_queue_.empty()) {
        event_queue_.pop();
    }
}

// MemoryPool implementation
MemoryPool::MemoryPool(size_t block_size, size_t initial_blocks)
    : current_block_(nullptr), block_size_(block_size) {

    for (size_t i = 0; i < initial_blocks; ++i) {
        allocate_new_block();
    }
}

MemoryPool::~MemoryPool() {
    Block* current = current_block_;
    while (current) {
        Block* next = current->next;
        std::free(current->data);
        delete current;
        current = next;
    }
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);

    if (!current_block_ || current_block_->used + size > current_block_->size) {
        allocate_new_block();
    }

    void* ptr = static_cast<char*>(current_block_->data) + current_block_->used;
    current_block_->used += size;

    allocated_bytes_ += size;
    total_allocations_++;

    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    // Simple implementation - in a real pool, we'd track allocations
    allocated_bytes_ -= size;
}

void MemoryPool::allocate_new_block() {
    Block* new_block = new Block;
    new_block->size = block_size_;
    new_block->used = 0;
    new_block->data = std::malloc(block_size_);
    new_block->next = current_block_;
    current_block_ = new_block;

    if (!new_block->data) {
        throw std::bad_alloc();
    }
}

// CSPRuntime implementation
CSPRuntime::CSPRuntime()
    : memory_pool_(4096, 10), max_threads_(std::thread::hardware_concurrency()),
      memory_pool_size_(1024 * 1024) { // 1MB default
}

CSPRuntime::~CSPRuntime() {
    stop();
}

void CSPRuntime::run_graph(Graph& graph, TimePoint start_time, TimePoint end_time) {
    std::lock_guard<std::mutex> lock(runtime_mutex_);

    running_ = true;
    stop_requested_ = false;

    auto start = std::chrono::high_resolution_clock::now();

    try {
        graph.run(start_time, end_time);

        auto end = std::chrono::high_resolution_clock::now();
        stats_.total_execution_time = std::chrono::duration_cast<Duration>(end - start);
        stats_.memory_allocated = memory_pool_.allocated_bytes();

    } catch (const std::exception& e) {
        running_ = false;
        throw;
    }

    running_ = false;
}

void CSPRuntime::run_graph_realtime(Graph& graph, Duration step_interval) {
    std::unique_lock<std::mutex> lock(runtime_mutex_);

    running_ = true;
    stop_requested_ = false;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (running_ && !stop_requested_) {
        if (paused_) {
            runtime_cv_.wait(lock, [this] { return !paused_ || stop_requested_; });
            continue;
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        auto end_time = current_time + step_interval;

        lock.unlock();
        try {
            execute_graph_step(graph, current_time);
        } catch (const std::exception& e) {
            std::cerr << "Runtime error: " << e.what() << std::endl;
            running_ = false;
            break;
        }
        lock.lock();

        // Wait for next step
        runtime_cv_.wait_until(lock, current_time + step_interval);
    }

    running_ = false;
}

void CSPRuntime::stop() {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    stop_requested_ = true;
    running_ = false;
    runtime_cv_.notify_all();
}

void CSPRuntime::pause() {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    paused_ = true;
}

void CSPRuntime::resume() {
    std::lock_guard<std::mutex> lock(runtime_mutex_);
    paused_ = false;
    runtime_cv_.notify_all();
}

void CSPRuntime::execute_graph_step(Graph& graph, TimePoint current_time) {
    // Simple single-threaded execution for now
    // In a full implementation, this would use the event scheduler
    // and potentially multiple threads

    // This is a simplified version - the actual graph execution
    // is handled by the Graph class itself
    stats_.total_nodes_executed++;
}

void CSPRuntime::process_node(BaseNode* node, TimePoint current_time) {
    try {
        node->process(current_time);
        stats_.total_events_processed++;
    } catch (const std::exception& e) {
        std::cerr << "Error processing node " << node->get_name() << ": " << e.what() << std::endl;
        throw;
    }
}

// Convenience function
void run(Graph& graph, TimePoint start_time, TimePoint end_time) {
    if (end_time == TimePoint{}) {
        end_time = start_time + std::chrono::seconds(1);
    }

    CSPRuntime runtime;
    runtime.run_graph(graph, start_time, end_time);
}

} // namespace core
} // namespace tsd