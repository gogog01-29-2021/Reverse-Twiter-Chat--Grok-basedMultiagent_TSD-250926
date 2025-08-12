import uuid

from .. import models
from .. import service_layer
from . import schemas


def get_node_groups() -> str:
    """Retrieves node groups.

    Returns:
        str: JSON string representation of node groups

    """
    node_groups: list[models.NodeGroup] = service_layer.get_node_groups()
    return schemas.NodeGroups.from_node_groups(node_groups).model_dump_json()


def create_workflow(node_ids: list[str]) -> dict:
    """Creates a workflow with given node IDs.

    Args:
        nodes: list[uuid.UUID] of the nodes to create a workflow

    Returns:
        uuid.UUID: ID of the created workflow
    """
    workflow_id: uuid.UUID = uuid.uuid4()
    workflow_name: str = f"Workflow {len(node_ids)} nodes"
    nodes: list[models.Node] = [service_layer.get_node(uuid.UUID(node_id)) for node_id in node_ids]
    workflow: models.Workflow = models.Workflow(
        id=workflow_id,
        name=workflow_name,
        nodes=nodes
    )
    service_layer.create_workflow(workflow)
    return {"work_flow_id": str(workflow.id)}


def get_workflow(workflow_id: uuid.UUID) -> str:
    """Retrieves a workflow.

    Args:
        workflow_id: uuid.UUID of the workflow to retrieve

    Returns:
        str: JSON string representation of the retrieved workflow
    """
    workflow: models.Workflow = service_layer.get_workflow(workflow_id)
    return workflow.model_dump_json()


def update_workflow(workflow_id: uuid.UUID, workflow: schemas.CreateWorkflow):
    """Updates a workflow.

    Args:
        workflow_id: uuid.UUID of the workflow to update
        workflow: schemas.WorkflowCreate

    Returns:
        None
    """
    workflow: models.Workflow = models.Workflow(
        id=workflow_id,
        name=workflow.name,
        nodes=[service_layer.get_node(node_id) for node_id in workflow.nodes]
    )
    service_layer.update_workflow(workflow_id, workflow)
