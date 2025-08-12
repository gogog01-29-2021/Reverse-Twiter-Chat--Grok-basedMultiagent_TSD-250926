from fastapi import APIRouter, HTTPException, Header, status
from pydantic import BaseModel, Field

# ===============================================================================
#  Request & Response Models
# ===============================================================================

class ResponseModel(BaseModel):
    pass

class Token(ResponseModel):
    """사용자 식별용 임시 토큰 발급 응답 모델"""
    access_token: str
    token_type: str = "bearer"

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "1234567890",
                "token_type": "bearer"
            }
        }


router = APIRouter(prefix="/api/v1", tags=["Authentication"])


@router.post(
    "/auth/token",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="사용자 식별용 임시 토큰 발급",
    description="사용자를 식별하기 위한 고유한 임시 토큰을 생성하고 반환합니다. 이 토큰을 사용하여 다른 API들을 호출할 수 있습니다."
)
async def sign_in():
    """
    사용자를 식별하기 위한 고유한 임시 토큰을 생성하고 반환합니다.
    이 토큰을 사용하여 다른 API들을 호출할 수 있습니다.
    """
    return Token(access_token="1234567890")