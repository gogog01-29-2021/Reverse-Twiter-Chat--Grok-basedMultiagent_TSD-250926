#pragma once

#include "node.hpp"
#include "types/base.hpp"
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <algorithm>

namespace tsd {
namespace core {

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    // Node management
    template<typename NodeType, typename... Args>
    NodeId add_node(Args&&... args) {
        NodeId id = next_node_id_++;
        auto node = std::make_unique<NodeType>(id, std::forward<Args>(args)...);
        nodes_[id] = std::move(node);
        return id;
    }

    // Connect nodes
    void connect(NodeId from, NodeId to,
                const std::string& from_output = "output",
                const std::string& to_input = "input");

    // Graph execution
    void run(TimePoint start_time, TimePoint end_time, Duration step = std::chrono::milliseconds(1));

    // Get node by ID
    BaseNode* get_node(NodeId id);
    const BaseNode* get_node(NodeId id) const;

    // Graph analysis
    std::vector<NodeId> topological_sort() const;
    bool has_cycles() const;
    void validate() const;

private:
    NodeId next_node_id_ = 1;
    std::unordered_map<NodeId, std::unique_ptr<BaseNode>> nodes_;
    std::vector<Edge> edges_;

    // Execution helpers
    void execute_step(TimePoint current_time);
    void propagate_data();

    // Topology helpers
    std::unordered_map<NodeId, std::vector<NodeId>> build_adjacency_list() const;
    std::unordered_map<NodeId, int> calculate_in_degrees() const;
};

// Graph builder helper class
class GraphBuilder {
public:
    GraphBuilder() : graph_(std::make_unique<Graph>()) {}

    template<typename NodeType, typename... Args>
    NodeId add_node(Args&&... args) {
        return graph_->add_node<NodeType>(std::forward<Args>(args)...);
    }

    GraphBuilder& connect(NodeId from, NodeId to,
                         const std::string& from_output = "output",
                         const std::string& to_input = "input") {
        graph_->connect(from, to, from_output, to_input);
        return *this;
    }

    std::unique_ptr<Graph> build() {
        graph_->validate();
        return std::move(graph_);
    }

private:
    std::unique_ptr<Graph> graph_;
};

} // namespace core
} // namespace tsd