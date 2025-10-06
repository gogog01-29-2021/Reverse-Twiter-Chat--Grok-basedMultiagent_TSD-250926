#include "node.hpp"
#include <iostream>
#include <stdexcept>

namespace tsd {
namespace core {

BaseNode::BaseNode(NodeId id, const std::string& name)
    : id_(id), name_(name.empty() ? "node_" + std::to_string(id) : name) {
}

void BaseNode::set_input(const std::string& name, const std::any& value) {
    inputs_[name] = value;
}

std::any BaseNode::get_output(const std::string& name) const {
    auto it = outputs_.find(name);
    if (it == outputs_.end()) {
        throw std::runtime_error("Output '" + name + "' not found in node " + name_);
    }
    return it->second;
}

bool BaseNode::has_input(const std::string& name) const {
    return inputs_.find(name) != inputs_.end();
}

bool BaseNode::all_inputs_ready() const {
    for (const auto& required : required_inputs_) {
        if (!has_input(required)) {
            return false;
        }
    }
    return true;
}

void BaseNode::register_input(const std::string& name) {
    required_inputs_.push_back(name);
}

void BaseNode::register_output(const std::string& name) {
    outputs_[name] = std::any{};
}

void BaseNode::set_output(const std::string& name, const std::any& value) {
    outputs_[name] = value;
}

} // namespace core
} // namespace tsd