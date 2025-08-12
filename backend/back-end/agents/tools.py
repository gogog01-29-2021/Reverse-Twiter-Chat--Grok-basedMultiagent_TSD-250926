from google.adk.tools import FunctionTool
from google.adk.agents.callback_context import CallbackContext

from ..node.entry_point.function_tools import create_workflow, get_node_groups


def insert_node_groups_to_state(callback_context: CallbackContext):
    """
    Inserts node groups to the callback context state.
    """
    node_groups: str = get_node_groups()
    callback_context.state["node_groups"] = node_groups


workflow_creation_tool = FunctionTool(create_workflow)
