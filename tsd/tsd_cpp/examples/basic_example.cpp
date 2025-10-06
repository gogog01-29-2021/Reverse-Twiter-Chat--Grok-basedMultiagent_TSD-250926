#include "core/engine/graph.hpp"
#include "core/engine/runtime.hpp"
#include "core/engine/node.hpp"
#include <iostream>
#include <chrono>

using namespace tsd::core;

int main() {
    std::cout << "=== TSD C++ Basic Example ===" << std::endl;

    // Create graph builder
    GraphBuilder builder;

    // Add nodes (equivalent to Python CSP example)
    auto const1_id = builder.add_node<ConstNode<int>>(1, "x");
    auto const2_id = builder.add_node<ConstNode<int>>(2, "y");
    auto add_id = builder.add_node<AddNode<int>>("sum");
    auto print1_id = builder.add_node<PrintNode<int>>("x: ", "print_x");
    auto print2_id = builder.add_node<PrintNode<int>>("y: ", "print_y");
    auto print_sum_id = builder.add_node<PrintNode<int>>("sum: ", "print_sum");

    // Connect nodes
    builder.connect(const1_id, add_id, "output", "lhs")
           .connect(const2_id, add_id, "output", "rhs")
           .connect(const1_id, print1_id)
           .connect(const2_id, print2_id)
           .connect(add_id, print_sum_id);

    // Build the graph
    auto graph = builder.build();

    // Execute the graph
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::seconds(1);

    std::cout << "Running graph..." << std::endl;

    try {
        run(*graph, start_time, end_time);
        std::cout << "Graph execution completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error running graph: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== Basic Example Complete ===" << std::endl;
    return 0;
}