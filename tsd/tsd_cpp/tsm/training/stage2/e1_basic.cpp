#include "core/engine/graph.hpp"
#include "core/engine/runtime.hpp"
#include "core/engine/node.hpp"
#include <iostream>
#include <chrono>

using namespace tsd::core;

// Equivalent to Python CSP e1_basic.py
int main() {
    std::cout << "=== TSM Stage 2 - Basic Example ===" << std::endl;

    // Create graph builder
    GraphBuilder builder;

    // Add nodes (equivalent to @csp.node add function)
    auto x_id = builder.add_node<ConstNode<int>>(1, "x");
    auto y_id = builder.add_node<ConstNode<int>>(2, "y");
    auto sum_id = builder.add_node<AddNode<int>>("sum");

    // Print nodes
    auto print_x_id = builder.add_node<PrintNode<int>>("x", "print_x");
    auto print_y_id = builder.add_node<PrintNode<int>>("y", "print_y");
    auto print_sum_id = builder.add_node<PrintNode<int>>("sum", "print_sum");

    // Connect nodes (equivalent to my_graph connections)
    builder.connect(x_id, sum_id, "output", "lhs")
           .connect(y_id, sum_id, "output", "rhs")
           .connect(x_id, print_x_id)
           .connect(y_id, print_y_id)
           .connect(sum_id, print_sum_id);

    // Build the graph
    auto graph = builder.build();

    // Execute (equivalent to csp.run(my_graph, starttime=datetime.now()))
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::milliseconds(100);

    std::cout << "Running CSP graph..." << std::endl;

    try {
        run(*graph, start_time, end_time);
        std::cout << "CSP graph completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error running CSP graph: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== TSM Basic Example Complete ===" << std::endl;
    return 0;
}