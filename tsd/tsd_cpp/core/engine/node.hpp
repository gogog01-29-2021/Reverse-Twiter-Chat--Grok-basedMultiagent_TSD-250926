#pragma once

#include "types/base.hpp"
#include <functional>
#include <unordered_map>
#include <any>
#include <typeindex>
#include <memory>
#include <iostream>

namespace tsd {
namespace core {

// Base class for all CSP nodes
class BaseNode {
public:
    BaseNode(NodeId id, const std::string& name = "");
    virtual ~BaseNode() = default;

    NodeId get_id() const { return id_; }
    const std::string& get_name() const { return name_; }

    // Node lifecycle
    virtual void setup() {}
    virtual void process(TimePoint current_time) = 0;
    virtual void teardown() {}

    // Input/output management
    void set_input(const std::string& name, const std::any& value);
    std::any get_output(const std::string& name) const;

    // Input availability check
    bool has_input(const std::string& name) const;
    bool all_inputs_ready() const;

protected:
    NodeId id_;
    std::string name_;
    std::unordered_map<std::string, std::any> inputs_;
    std::unordered_map<std::string, std::any> outputs_;
    std::vector<std::string> required_inputs_;

    void register_input(const std::string& name);
    void register_output(const std::string& name);
    void set_output(const std::string& name, const std::any& value);
};

// Templated node for type safety
template<typename InputType, typename OutputType>
class Node : public BaseNode {
public:
    Node(NodeId id, const std::string& name = "") : BaseNode(id, name) {
        register_input("input");
        register_output("output");
    }

    virtual OutputType compute(const TimeSeries<InputType>& input) = 0;

    void process(TimePoint current_time) override {
        if (!has_input("input")) return;

        auto input_any = get_input_any("input");
        try {
            if (!input_any.has_value()) {
                throw std::runtime_error("Input not set for node " + name_);
            }
            auto input_ts = std::any_cast<TimeSeries<InputType>>(input_any);
            auto result = compute(input_ts);
            set_output("output", TimeSeries<OutputType>{current_time, result});
            inputs_.erase("input");
        } catch (const std::bad_any_cast& e) {
            // Handle type mismatch
            throw std::runtime_error("Type mismatch in node " + name_ + ": " + e.what());
        }
    }

private:
    std::any get_input_any(const std::string& name) const {
        auto it = inputs_.find(name);
        return (it != inputs_.end()) ? it->second : std::any{};
    }
};

// Common node implementations

// Constant value node
template<typename T>
class ConstNode : public Node<void, T> {
private:
    T value_;

public:
    ConstNode(NodeId id, T value, const std::string& name = "const")
        : Node<void, T>(id, name), value_(value) {}

    T compute(const TimeSeries<void>&) override {
        return value_;
    }
};

// Mathematical operation nodes
template<typename T>
class AddNode : public BaseNode {
private:
    T last_result_{};

public:
    AddNode(NodeId id, const std::string& name = "add") : BaseNode(id, name) {
        register_input("lhs");
        register_input("rhs");
        register_output("output");
        required_inputs_ = {"lhs", "rhs"};
    }

    void process(TimePoint current_time) override {
        if (!all_inputs_ready()) return;

        try {
            auto lhs = std::any_cast<TimeSeries<T>>(inputs_["lhs"]);
            auto rhs = std::any_cast<TimeSeries<T>>(inputs_["rhs"]);

            last_result_ = lhs.value + rhs.value;
            set_output("output", TimeSeries<T>{current_time, last_result_});
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Type mismatch in AddNode: " + std::string(e.what()));
        }
    }
};

// Accumulation node
template<typename T>
class AccumNode : public Node<T, T> {
private:
    T accumulated_{};

public:
    AccumNode(NodeId id, const std::string& name = "accum")
        : Node<T, T>(id, name) {}

    T compute(const TimeSeries<T>& input) override {
        accumulated_ += input.value;
        return accumulated_;
    }
};

// Print node for debugging
template<typename T>
class PrintNode : public Node<T, T> {
private:
    std::string prefix_;

public:
    PrintNode(NodeId id, const std::string& prefix = "", const std::string& name = "print")
        : Node<T, T>(id, name), prefix_(prefix) {}

    T compute(const TimeSeries<T>& input) override {
        std::cout << prefix_ << input.value << " @ "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         input.timestamp.time_since_epoch()
                     ).count() << "ms" << std::endl;
        return input.value;
    }
};

} // namespace core
} // namespace tsd