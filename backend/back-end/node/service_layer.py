import uuid
from . import models
from .repository import NodeRepository

def get_node(node_id: uuid.UUID) -> models.Node:
    node_repository = NodeRepository()
    return node_repository.get_node(node_id)

def get_node_groups() -> list[models.NodeGroup]:
    node_repository = NodeRepository()
    return node_repository.get_node_groups()

def create_workflow(workflow: models.Workflow) -> models.Workflow:
    node_repository = NodeRepository()
    return node_repository.add_workflow(workflow)

def get_workflows() -> list[models.Workflow]:
    node_repository = NodeRepository()
    return node_repository.get_workflows()

def get_workflow(workflow_id: uuid.UUID) -> models.Workflow:
    node_repository = NodeRepository()
    return node_repository.get_workflow(workflow_id)

def update_workflow(workflow_id: uuid.UUID, workflow: models.Workflow) -> models.Workflow:
    node_repository = NodeRepository()
    return node_repository.update_workflow(workflow_id, workflow)