from typing import TypedDict, Literal, Optional

class BinanceSpotOrderRequest(TypedDict):
    symbol: str  # 거래쌍 심볼 (예: "BTCUSDT")
    side: Literal["BUY", "SELL"]  # 주문 방향: BUY(매수), SELL(매도)
    type: Literal[
        "LIMIT",             # 지정가 주문
        "MARKET",            # 시장가 주문
        "STOP_LOSS",         # 스탑로스 (지정된 가격 도달 시 시장가로 손절)
        "STOP_LOSS_LIMIT",   # 스탑로스 + 지정가 (트리거 후 지정가로 손절)
        "TAKE_PROFIT",       # 이익 실현 (시장가)
        "TAKE_PROFIT_LIMIT", # 이익 실현 (지정가)
        "LIMIT_MAKER"        # 지정가 주문 (Maker-only, Taker 수수료 방지)
    ]
    timeInForce: Optional[Literal["GTC", "IOC", "FOK"]]  
    # 주문 유효 기간: 
    # - GTC: Good-Til-Canceled (취소 전까지 유효)
    # - IOC: Immediate-Or-Cancel (즉시 체결, 나머지 취소)
    # - FOK: Fill-Or-Kill (전량 즉시 체결 안 되면 취소)

    quantity: Optional[str]  # 주문 수량 (Base 자산 기준, 예: BTC 수량)
    quoteOrderQty: Optional[str]  
    # 주문 금액 (Quote 자산 기준, 예: USDT로 100달러어치 매수)
    # MARKET 주문에 사용

    price: Optional[str]  # 주문 가격 (LIMIT 주문 시 필수)

    newClientOrderId: Optional[str]  
    # 사용자 정의 주문 ID (중복 방지용, 없으면 자동 생성)

    stopPrice: Optional[str]  
    # 스탑 주문 트리거 가격 (STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT에 사용)

    icebergQty: Optional[str]  
    # Iceberg 주문 공개 수량 (나머지 숨김, 예: 대량 주문 숨기기)

    newOrderRespType: Optional[Literal["ACK", "RESULT", "FULL"]]  
    # 주문 응답 타입:
    # - ACK: 주문 수신만 확인 (빠름)
    # - RESULT: 주문 처리 결과 반환
    # - FULL: 전체 주문 체결 내역까지 반환

    recvWindow: Optional[int]  
    # 요청 유효 시간(ms), 서버 타임과의 시간차에 대한 허용 범위 (예: 5000)

    trailingDelta: Optional[int]  
    # 트레일링 스탑 주문에서 가격 차이 (단위: 가격 단위)

    strategyId: Optional[int]  
    # 거래 전략 ID (선택적, 알고리즘 트레이딩 시 사용)

    strategyType: Optional[int]  
    # 거래 전략 유형 (선택적)

    selfTradePreventionMode: Optional[Literal["NONE", "EXPIRE_TAKER", "EXPIRE_MAKER", "EXPIRE_BOTH"]]  
    # 자기체결 방지 모드:
    # - NONE: 비활성화 (기본값)
    # - EXPIRE_TAKER: Taker 주문 취소
    # - EXPIRE_MAKER: Maker 주문 취소
    # - EXPIRE_BOTH: 양쪽 주문 모두 취소

    goodTillDate: Optional[int]  
    # GTD (Good-Till-Date) 주문 만료 시간 (밀리초 Epoch Time)

    timestamp: int  # 서버 시간 동기화용 타임스탬프 (밀리초 Epoch Time, 필수)

    signature: str  # HMAC SHA256으로 생성한 서명 (API Secret 이용, 필수)