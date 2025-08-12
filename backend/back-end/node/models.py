import uuid
from pydantic import BaseModel

type NodeName = str
type NodeDescription = str

class Node(BaseModel):
    id: uuid.UUID
    name: NodeName
    description: NodeDescription
    # icon: str
    # type: str
    # inputs: List[str]
    # outputs: List[str]

type NodeGroupName = str

class NodeGroup(BaseModel):
    name: NodeGroupName
    nodes: list[Node]

type WorkflowName = str

class Workflow(BaseModel):
    id: uuid.UUID
    name: WorkflowName
    nodes: list[Node]
