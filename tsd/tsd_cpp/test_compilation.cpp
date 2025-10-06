// Simple test to verify C++ compilation works
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "TSD C++ Migration Test Compilation" << std::endl;
    std::cout << "===================================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Test basic C++20 features
    auto lambda = []() -> std::string {
        return "C++20 lambdas working!";
    };

    std::cout << lambda() << std::endl;

    // Test threading
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "Threading support working!" << std::endl;
    });
    t.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Test completed in " << duration.count() << "ms" << std::endl;
    std::cout << "✅ C++ environment is ready for TSD migration!" << std::endl;

    return 0;
}