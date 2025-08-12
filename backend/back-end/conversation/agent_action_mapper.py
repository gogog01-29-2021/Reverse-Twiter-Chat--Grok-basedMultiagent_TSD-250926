agent_action_mapper: dict[str, str] = {
    "create_workflow": "work_flow",
}

def get_agent_action_type(key: str) -> str:
    return agent_action_mapper.get(key, key)