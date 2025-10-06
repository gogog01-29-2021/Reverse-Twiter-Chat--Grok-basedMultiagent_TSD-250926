from typing import Literal, Optional, TypedDict

class OKXSpotOrderRequest(TypedDict):
    instId: str  
    # 거래쌍 심볼 (예: "BTC-USDT")
    
    tdMode: Literal["cash"]  
    # 거래 모드: 현물 거래는 항상 "cash"로 설정 (마진 거래는 다른 값 사용)
    
    side: Literal["buy", "sell"]  
    # 주문 방향: buy(매수), sell(매도)
    
    ordType: Literal[
        "limit",              # 지정가 주문
        "market",             # 시장가 주문
        "post_only",          # Post Only (Maker 주문만, Taker 수수료 방지)
        "fok",                 # Fill-Or-Kill (즉시 전량 체결 안 되면 주문 취소)
        "ioc",                 # Immediate-Or-Cancel (즉시 체결, 나머지 취소)
        "optimal_limit_ioc"    # 시장가 주문과 유사, 최적 가격으로 IOC 실행 (OKX 전용)
    ]
    # 주문 유형: 체결 조건에 따라 설정
    
    sz: str  
    # 주문 수량 (Base 자산 기준, 예: BTC 수량)

    px: Optional[str]  
    # 주문 가격 (ordType이 "limit", "post_only"일 때 필수)

    clOrdId: Optional[str]  
    # 사용자 정의 주문 ID (최대 64자, 주문 추적/관리용)

    tag: Optional[str]  
    # 사용자 정의 태그 (주문 그룹화, 로깅 목적)

    reduceOnly: Optional[bool]  
    # 포지션을 줄이기 위한 주문 여부 (현물에서는 보통 False)

    tgtCcy: Optional[Literal["base_ccy", "quote_ccy"]]  
    # 시장가 주문 시 결제 기준 통화:
    # - base_ccy: 수량 기준 (예: BTC 0.1개)
    # - quote_ccy: 금액 기준 (예: USDT 100달러어치 구매)
    # - 예: tgtCcy="quote_ccy" → quote 통화 기준으로 주문 금액 지정

    ccy: Optional[str]  
    # 통화 코드 (주로 마켓 주문에서 사용, 예: "USDT")

    quickMgnType: Optional[Literal["manual", "auto_borrow", "auto_repay"]]  
    # 마진 거래 관련 설정:
    # - manual: 수동 마진
    # - auto_borrow: 부족 시 자동 대출
    # - auto_repay: 주문 완료 시 자동 상환
    # - 현물 거래에서는 일반적으로 사용하지 않음

    stpId: Optional[str]  
    # Self-Trade Prevention을 위한 사용자 정의 식별 ID (중복 매매 방지)

    stpMode: Optional[Literal["cancel_maker", "cancel_taker", "cancel_both"]]  
    # 자기체결 방지 모드:
    # - cancel_maker: Maker 주문 취소
    # - cancel_taker: Taker 주문 취소
    # - cancel_both: 두 주문 모두 취소

    cancelAfter: Optional[str]  
    # 지정 시간(ms) 후 주문 자동 취소 (GTD 유사 기능, 예: "60000" → 60초 후 자동 취소)