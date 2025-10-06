#include "graph.hpp"
#include <stdexcept>
#include <unordered_set>
#include <iostream>

namespace tsd {
namespace core {

void Graph::connect(NodeId from, NodeId to, const std::string& from_output, const std::string& to_input) {
    // Validate nodes exist
    if (nodes_.find(from) == nodes_.end()) {
        throw std::runtime_error("Source node " + std::to_string(from) + " not found");
    }
    if (nodes_.find(to) == nodes_.end()) {
        throw std::runtime_error("Target node " + std::to_string(to) + " not found");
    }

    edges_.push_back({from, to, from_output, to_input});
}

void Graph::run(TimePoint start_time, TimePoint end_time, Duration step) {
    validate();

    auto current_time = start_time;

    // Setup all nodes
    for (auto& [id, node] : nodes_) {
        node->setup();
    }

    std::cout << "Starting graph execution from "
              << std::chrono::duration_cast<std::chrono::milliseconds>(start_time.time_since_epoch()).count()
              << "ms to "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time.time_since_epoch()).count()
              << "ms" << std::endl;

    // Main execution loop
    while (current_time <= end_time) {
        execute_step(current_time);
        propagate_data();
        current_time += step;
    }

    // Teardown all nodes
    for (auto& [id, node] : nodes_) {
        node->teardown();
    }
}

BaseNode* Graph::get_node(NodeId id) {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

const BaseNode* Graph::get_node(NodeId id) const {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

std::vector<NodeId> Graph::topological_sort() const {
    auto adj_list = build_adjacency_list();
    auto in_degrees = calculate_in_degrees();

    std::queue<NodeId> queue;
    std::vector<NodeId> result;

    // Start with nodes that have no incoming edges
    for (const auto& [node_id, node] : nodes_) {
        if (in_degrees[node_id] == 0) {
            queue.push(node_id);
        }
    }

    while (!queue.empty()) {
        NodeId current = queue.front();
        queue.pop();
        result.push_back(current);

        // Process all neighbors
        auto it = adj_list.find(current);
        if (it != adj_list.end()) {
            for (NodeId neighbor : it->second) {
                in_degrees[neighbor]--;
                if (in_degrees[neighbor] == 0) {
                    queue.push(neighbor);
                }
            }
        }
    }

    return result;
}

bool Graph::has_cycles() const {
    auto sorted = topological_sort();
    return sorted.size() != nodes_.size();
}

void Graph::validate() const {
    if (has_cycles()) {
        throw std::runtime_error("Graph contains cycles");
    }

    // Validate all edges reference valid nodes and ports
    for (const auto& edge : edges_) {
        auto from_node = get_node(edge.from);
        auto to_node = get_node(edge.to);

        if (!from_node) {
            throw std::runtime_error("Edge references invalid source node: " + std::to_string(edge.from));
        }
        if (!to_node) {
            throw std::runtime_error("Edge references invalid target node: " + std::to_string(edge.to));
        }
    }
}

void Graph::execute_step(TimePoint current_time) {
    auto execution_order = topological_sort();

    for (NodeId node_id : execution_order) {
        auto node = get_node(node_id);
        if (node) {
            try {
                node->process(current_time);
            } catch (const std::exception& e) {
                std::cerr << "Error processing node " << node->get_name()
                          << " (ID: " << node_id << "): " << e.what() << std::endl;
                throw;
            }
        }
    }
}

void Graph::propagate_data() {
    for (const auto& edge : edges_) {
        auto from_node = get_node(edge.from);
        auto to_node = get_node(edge.to);

        if (from_node && to_node) {
            try {
                auto output_data = from_node->get_output(edge.from_output);
                if (!output_data.has_value()) {
                    continue;
                }
                to_node->set_input(edge.to_input, output_data);
            } catch (const std::exception& e) {
                std::cerr << "Error propagating data from node " << edge.from
                          << " to node " << edge.to << ": " << e.what() << std::endl;
                // Continue with other edges
            }
        }
    }
}

std::unordered_map<NodeId, std::vector<NodeId>> Graph::build_adjacency_list() const {
    std::unordered_map<NodeId, std::vector<NodeId>> adj_list;

    for (const auto& edge : edges_) {
        adj_list[edge.from].push_back(edge.to);
    }

    return adj_list;
}

std::unordered_map<NodeId, int> Graph::calculate_in_degrees() const {
    std::unordered_map<NodeId, int> in_degrees;

    // Initialize all nodes with 0 in-degree
    for (const auto& [node_id, node] : nodes_) {
        in_degrees[node_id] = 0;
    }

    // Count incoming edges
    for (const auto& edge : edges_) {
        in_degrees[edge.to]++;
    }

    return in_degrees;
}

} // namespace core
} // namespace tsd