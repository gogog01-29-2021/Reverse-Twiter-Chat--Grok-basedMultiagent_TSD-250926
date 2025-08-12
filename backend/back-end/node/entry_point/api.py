import uuid

from fastapi import APIRouter

from .. import models
from .. import service_layer as service
from . import schemas


# ===============================================================================
#  FastAPI 라우터
# ===============================================================================

router = APIRouter(prefix="/api/v1")


@router.get(
    "/nodes",
    response_model=schemas.NodeGroups,
    summary="워크플로우 구성 요소(노드) 조회",
    tags=["Nodes"],
)
async def get_available_nodes():
    """
    사용자가 워크플로우를 구성할 때 사용할 수 있는 모든 노드의 목록을 반환합니다.
    """
    node_groups: models.NodeGroups = service.get_node_groups()
    return schemas.NodeGroups.from_node_groups(node_groups)


@router.post(
    "/workflows/",
    summary="워크플로우 저장",
    response_model=schemas.WorkflowDetail,
    tags=["Workflows"],
)
async def save_workflow(
    workflow_data: schemas.CreateWorkflow,
):
    workflow = models.Workflow(
        id=uuid.uuid4(),
        name=workflow_data.name,
        nodes=[service.get_node(node_id) for node_id in workflow_data.nodes]
    )
    service.create_workflow(workflow)
    return schemas.WorkflowDetail.from_workflow(workflow)


@router.get(
    "/workflows",
    summary="워크플로우 목록 조회",
    response_model=schemas.WorkflowList,
    tags=["Workflows"],
)
async def get_workflows():
    workflows: list[models.Workflow] = service.get_workflows()
    return schemas.WorkflowList.from_workflows(workflows)


@router.get(
    "/workflows/{workflow_id}",
    summary="워크플로우 상세 조회",
    response_model=schemas.WorkflowDetail,
    tags=["Workflows"],
)
async def get_workflow(workflow_id: uuid.UUID):
    workflow: models.Workflow = service.get_workflow(workflow_id)
    return schemas.WorkflowDetail.from_workflow(workflow)

