import uuid
from pydantic import BaseModel

from .. import models

class RequestModel(BaseModel):
    pass

class ResponseModel(BaseModel):
    pass

class NodeSummary(ResponseModel):
    id: uuid.UUID
    name: str
    description: str
    # icon: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "CAD 불러오기",
                "description": "This is a node"
            }
        }

    @classmethod
    def from_node(cls, node: models.Node) -> "NodeSummary":
        """Node 모델을 NodeSummary로 변환"""
        return cls(
            id=node.id,
            name=node.name,
            description=node.description
        )


class NodeGroup(ResponseModel):
    name: str
    nodes: list[NodeSummary]

    class Config:
        json_schema_extra = {
            "example": {
                "name": "전처리",
                "nodes": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "CAD 불러오기",
                        "description": "This is a node"
                    }
                ]
            }
        }

    @classmethod
    def from_node_group(cls, node_group: models.NodeGroup) -> "NodeGroup":
        """NodeGroup 모델을 NodeGroupResponse로 변환"""
        return cls(
            name=node_group.name,
            nodes=[NodeSummary.from_node(node) for node in node_group.nodes]
        )


class NodeGroups(ResponseModel):
    groups: list[NodeGroup]

    class Config:
        json_schema_extra = {
            "example": {
                "groups": [
                    {
                        "name": "전처리",
                        "nodes": [
                            {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "name": "CAD 불러오기",
                                "description": "This is a node"
                            }
                        ]
                    }
                ]
            }
        }

    @classmethod
    def from_node_groups(cls, node_groups: list[models.NodeGroup]) -> "NodeGroups":
        """NodeGroup 리스트를 NodeGroupsResponse로 변환"""
        return cls(
            groups=[NodeGroup.from_node_group(group) for group in node_groups]
        )


# Workflow 관련 스키마들
class CreateWorkflow(RequestModel):
    name: str
    nodes: list[uuid.UUID]  # 노드의 ID 목록
    # links: List[WorkflowLink]

    class Config:
        json_schema_extra = {
            "example": {
                "name": "건물 지진 시뮬레이션",
                "nodes": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "123e4567-e89b-12d3-a456-426614174001",
                    "123e4567-e89b-12d3-a456-426614174002",
                ]
            }
        }


class WorkflowSummary(ResponseModel):
    id: uuid.UUID
    name: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "건물 지진 시뮬레이션"
            }
        }
    
    @classmethod
    def from_workflow(cls, workflow: models.Workflow) -> "WorkflowSummary":
        return cls(
            id=workflow.id,
            name=workflow.name
        )


class WorkflowDetail(ResponseModel):
    id: uuid.UUID
    name: str
    nodes: list[NodeSummary]  # 노드의 ID 목록

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "건물 지진 시뮬레이션",
                "nodes": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "CAD 불러오기",
                        "description": "This is a node"
                    },
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174001",
                        "name": "메싱",
                        "description": "This is a node"
                    },
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174002",
                        "name": "메시 품질 검사",
                        "description": "This is a node"
                    }
                ]
            }
        }
    
    @classmethod
    def from_workflow(cls, workflow: models.Workflow) -> "WorkflowDetail":
        return cls(
            id=workflow.id,
            name=workflow.name,
            nodes=[NodeSummary.from_node(node) for node in workflow.nodes]
        )


class WorkflowList(ResponseModel):
    workflows: list[WorkflowSummary]

    class Config:
        json_schema_extra = {
            "example": {
                "workflows": [
                    {"id": "123e4567-e89b-12d3-a456-426614174000", "name": "건물 지진 시뮬레이션"},
                    {"id": "34657890-e89b-12d3-a456-426614174001", "name": "해일 시뮬레이션"}
                ]
            }
        } 
    
    @classmethod
    def from_workflows(cls, workflows: list[models.Workflow]) -> "WorkflowList":
        return cls(
            workflows=[WorkflowSummary.from_workflow(workflow) for workflow in workflows]
        )