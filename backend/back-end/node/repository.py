import uuid

from .models import Node, NodeGroup, Workflow


inMemoryNodeDB: list[NodeGroup] = [
    NodeGroup(
        name="전처리",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="CAD 불러오기",
                description="STEP, IGES, BREP 등 표준 CAD 파일을 플랫폼으로 불러옵니다. (핵심 입력 기능)"
            ),
            Node(
                id=uuid.uuid4(),
                name="메싱",
                description="형상에 1D, 2D, 3D 격자(Mesh)를 생성합니다. (알고리즘: Netgen, GHS3D, Mefisto 등)"
            ),
            Node(
                id=uuid.uuid4(),
                name="메시 품질 검사",
                description="생성된 메시의 종횡비(Aspect Ratio), 왜곡(Skewness) 등 품질 지표를 검사하고 보고합니다."
            ),
        ]
    ),
    NodeGroup(
        name="솔버 설정 및 실행",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="물성치 정의",
                description="탄성계수, 밀도, 열전도율 등 시뮬레이션에 필요한 재료의 물성치를 입력합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="물리 모델 선택",
                description="해석할 물리 현상을 선택합니다. (예: 구조 정해석, 열전달, 비압축성 유동 등)"
            ),
            Node(
                id=uuid.uuid4(),
                name="경계 조건 설정",
                description="생성된 그룹에 하중, 구속, 온도, 유속 등 경계 조건을 할당합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="초기 조건 설정",
                description="과도 해석(Transient)을 위해 시스템의 초기 상태(온도, 압력 등)를 정의합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="OpenFOAM 솔버",
                description="유체 역학(CFD) 시뮬레이션을 실행합니다. (솔버 종류: simpleFoam, pisoFoam 등)"
            ),
            Node(
                id=uuid.uuid4(),
                name="Code_Aster 솔버",
                description="구조, 열, 진동 등 유한요소해석(FEA)을 실행합니다. (해석 종류: STAT_NON_LINE, THER_LINE 등)"
            ),
            Node(
                id=uuid.uuid4(),
                name="솔버 제어",
                description="계산 시간, 타임스텝, 수렴 조건 등 솔버의 상세 파라미터를 설정합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="연성 해석 설정",
                description="열-구조, 유체-구조 등 두 가지 이상의 물리 현상을 연동하여 해석하도록 설정합니다."
            ),
        ]
    ),
    NodeGroup(
        name="후처리 및 시각화",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="결과 불러오기",
                description="솔버가 생성한 결과 파일(.case, .foam 등)을 불러옵니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="컨투어 플롯",
                description="변형, 응력, 온도, 압력 분포 등을 색상으로 시각화합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="데이터 추출",
                description="최대/최소값, 평균값 등 특정 지표를 계산 결과로부터 추출합니다."
            ),
        ]
    ),
    NodeGroup(
        name="자동화 및 AI",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="파라미터 최적화",
                description="지정된 변수(치수, 온도 등)의 범위를 자동으로 변경하며 반복 계산을 수행합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="AI 결과 분석",
                description="자연어 질문을 통해 방대한 결과 데이터에서 의미 있는 패턴이나 핵심 요약을 도출합니다."
            ),
        ]
    ),
    NodeGroup(
        name="데이터 및 유틸리티",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="변수 설정",
                description="워크플로우 전체에서 사용할 수 있는 전역 변수(예: 기준 온도)를 정의합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="조건 분기",
                description="특정 조건의 참/거짓에 따라 워크플로우의 실행 경로를 분기합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="반복 실행",
                description="지정된 횟수만큼 워크플로우의 특정 부분을 반복적으로 실행합니다."
            ),
            Node(
                id=uuid.uuid4(),
                name="알림 보내기",
                description="계산 완료 또는 오류 발생 시 이메일, 슬랙 등으로 알림을 보냅니다."
            ),
        ]
    ),
    NodeGroup(
        name="내 스크립트",
        nodes=[
            Node(
                id=uuid.uuid4(),
                name="Python 스크립트",
                description="사용자가 작성한 Python 코드를 실행합니다. (Numpy, Pandas, Matplotlib 등 라이브러리 지원)"
            ),
            Node(
                id=uuid.uuid4(),
                name="Salome Python API",
                description="Salome의 기능을 제어하는 Python 스크립트를 직접 작성하고 실행합니다."
            ),
        ]
    ),
]


inMemoryWorkflowDB: dict[uuid.UUID, Workflow] = {}


class NodeRepository:
    def __init__(self):
        self.node_groups = inMemoryNodeDB
        self.workflows = inMemoryWorkflowDB
    
    def get_node(self, node_id: uuid.UUID) -> Node:
        for node_group in self.node_groups:
            for node in node_group.nodes:
                if node.id == node_id:
                    return node
        raise ValueError(f"Node with id {node_id} not found")

    def get_node_groups(self) -> list[NodeGroup]:
        return self.node_groups
    
    def get_workflows(self) -> list[Workflow]:
        return list(self.workflows.values())
    
    def get_workflow(self, workflow_id: uuid.UUID) -> Workflow:
        return self.workflows[workflow_id]
    
    def add_workflow(self, workflow: Workflow) -> Workflow:
        self.workflows[workflow.id] = workflow
        return workflow
    
    def update_workflow(self, workflow_id: uuid.UUID, workflow: Workflow) -> Workflow:
        self.workflows[workflow_id] = workflow
        return workflow
