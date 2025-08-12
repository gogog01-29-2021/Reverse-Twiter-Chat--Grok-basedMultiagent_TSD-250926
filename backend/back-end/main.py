import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .conversation.api import router as conversation_router
from .node.entry_point.api import router as node_router
from .auth.api import router as auth_router



api_description = """
# 통합 에이전트 API
CAE 과정 자동화를 위한 멀티 에이전트 시스템 및 플랫폼을 위한 API 문서입니다.
"""


tags_metadata = [
    {
        "name": "Interact with Agents",
        "description": """
## 📝 효율적인 실시간 상호작용을 위한 API 워크플로우

AI 응답은 비동기적으로 처리되므로, 아래의 **폴링(Polling) 워크플로우**를 따르길 추천합니다.

### **권장 워크플로우 (Recommended Workflow)**

**1. 초기 데이터 로드 (Initial Load)**
   - 사용자가 대화방에 처음 입장하면, `GET /conversations/{conversation_id}/messages`를 호출하여 이전 대화 내역을 모두 화면에 표시합니다.

**2. 메시지 전송 (Send Message)**
   - 사용자가 메시지를 보내면, `POST /conversations/{conversation_id}/messages`를 호출합니다.
   - 서버는 즉시 `202Accepted`와 함께 `{ "message_id": "..." }`를 반환합니다.
   - 프론트엔드는 이 `message_id`를 '현재 응답을 기다리는 메시지'로 상태에 저장해야 합니다.

**3. 상태 폴링 (Efficient Polling)**
   - 메시지 전송 직후, `setInterval` 등을 이용해 **`GET /messages/pending` API만 반복적으로 호출**합니다. **(주의: `GET .../messages`를 반복 호출하면 네트워크 부하가 발생합니다.)**
   - `pending` API의 응답 목록에서 내가 기다리던 `message_id`가 **사라지면**, AI의 처리가 완료된 것입니다.

**4. 최종 데이터 동기화 (Final Sync)**
   - 폴링이 끝나면, `GET /conversations/{id}/messages`를 **다시 한번 호출**하여 AI의 답변이 포함된 최신 대화 내역 전체를 가져와 화면을 갱신합니다.
""",
    },
    {
        "name": "Nodes",
        "description": "Nodes API",
    },
    {
        "name": "Workflows",
        "description": "Workflows API",
    },
    {
        "name": "Auth",
        "description": "Auth API",
    },
]


app = FastAPI(
    title="통합 에이전트 REST API",
    description=api_description,
    version="0.0.1",
    openapi_tags=tags_metadata,
)


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(conversation_router)
app.include_router(node_router)
app.include_router(auth_router)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Multi Agents API. Go to /docs to see the API documentation."}
