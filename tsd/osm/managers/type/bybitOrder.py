from typing import TypedDict, Literal, Optional

class BybitSpotOrderRequest(TypedDict):
    category: Literal["spot"]  # 거래 카테고리: spot 고정 (선물/옵션과 구분)
    
    symbol: str  # 거래쌍 심볼 (예: "BTCUSDT")
    
    side: Literal["Buy", "Sell"]  # 주문 방향: Buy(매수), Sell(매도)
    
    orderType: Literal["Limit", "Market"]  
    # 주문 유형: 
    # - Limit: 지정가 주문
    # - Market: 시장가 주문
    
    qty: str  # 주문 수량 (Base 자산 기준, 예: BTC 수량)
    
    price: Optional[str]  
    # 주문 가격 (Limit 주문 시 필수, Market 주문 시 생략)
    
    triggerPrice: Optional[str]  
    # 트리거 주문 사용 시 발동 가격 (예: 스탑로스, 테이크프로핏 설정 시)
    
    triggerBy: Optional[Literal["LastPrice", "IndexPrice", "MarkPrice"]]  
    # 트리거 기준 가격:
    # - LastPrice: 최종 체결 가격
    # - IndexPrice: 지수 가격
    # - MarkPrice: 마크 가격 (주로 파생상품에서 사용)
    
    orderFilter: Optional[Literal["Order", "tpslOrder"]]  
    # 주문 필터:
    # - "Order": 일반 주문
    # - "tpslOrder": 테이크프로핏/스탑로스 주문
    
    timeInForce: Optional[Literal["GTC", "IOC", "FOK", "PostOnly"]]  
    # 주문 유효 기간:
    # - GTC: Good-Til-Canceled (취소 전까지 유효)
    # - IOC: Immediate-Or-Cancel (즉시 체결, 나머지 취소)
    # - FOK: Fill-Or-Kill (전량 즉시 체결 안 되면 취소)
    # - PostOnly: Maker 주문만 허용 (수수료 절감 목적)
    
    orderLinkId: Optional[str]  
    # 클라이언트 정의 주문 ID (주문 추적 및 관리용, 중복 방지)
    
    isLeverage: Optional[int]  
    # 레버리지 사용 여부 (일부 경우 Spot에서도 필드 요구, 0 또는 1)
    
    takeProfit: Optional[str]  
    # 테이크 프로핏 가격 (목표가 도달 시 익절)
    
    stopLoss: Optional[str]  
    # 스탑로스 가격 (손절가 설정)
    
    tpTriggerBy: Optional[Literal["LastPrice", "IndexPrice", "MarkPrice"]]  
    # 테이크 프로핏 트리거 기준 가격 종류
    
    slTriggerBy: Optional[Literal["LastPrice", "IndexPrice", "MarkPrice"]]  
    # 스탑로스 트리거 기준 가격 종류
    
    reduceOnly: Optional[bool]  
    # 오픈 포지션을 줄이기 위한 주문 여부 (주로 파생상품, Spot에선 일반적으로 사용 안 함)
    
    closeOnTrigger: Optional[bool]  
    # 트리거 주문 발생 시 포지션 즉시 청산 (주로 파생상품 전용)
    
    mmp: Optional[bool]  
    # Market Maker Protection: 급격한 시장 변동으로 인한 손실 방지 기능
    
    tpslMode: Optional[Literal["Full", "Partial"]]  
    # TP/SL 동작 모드:
    # - Full: 전체 포지션 종료
    # - Partial: 일부 포지션 종료 (Bybit는 일부 주문에 대해 지원)
    
    positionIdx: Optional[int]  
    # 멀티 포지션 모드에서 포지션 인덱스 지정 (1: One-Way, 2: Buy, 3: Sell)