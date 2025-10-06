from typing import TypedDict, Literal, Optional
from core.types.trade import OrderSide

# -----------------------------
# Coinbase Order Configurations
# -----------------------------

class MarketMarketIOC(TypedDict):
    # 시장가 주문 (Immediate or Cancel)
    # 즉시 체결 가능한 부분만 체결하고 나머지는 취소
    quote_size: str  # Quote 통화 기준 주문 금액 (예: USD로 500달러어치 구매)
    base_size: str   # Base 통화 기준 주문 수량 (예: BTC 0.01개 구매)


class SorLimitIOC(TypedDict):
    # SOR (Smart Order Routing) + 지정가 + IOC 
    # 여러 거래소에서 가장 좋은 가격으로 즉시 체결, 나머지 취소
    quote_size: str  # 주문 총 금액 (quote 통화 기준)
    base_size: str   # 주문 수량 (base 통화 기준)
    limit_price: str # 지정가 (이 가격 이하/이상으로만 체결)


class LimitLimitGTC(TypedDict):
    # 일반적인 지정가 주문 (Good-Til-Cancelled)
    # 취소 전까지 계속 유효
    quote_size: str
    base_size: str
    limit_price: str # 체결 희망 가격
    post_only: bool  # True면 Maker 주문만 허용 (Taker 수수료 방지)


class LimitLimitGTD(TypedDict):
    # 지정가 + GTD (Good-Til-Date)
    # 특정 시간까지 유효, 이후 미체결 주문 자동 취소
    quote_size: str
    base_size: str
    limit_price: str
    end_time: str    # 주문 만료 시간 (ISO8601, UTC, 예: '2025-05-12T09:00:00Z')
    post_only: bool


class LimitLimitFOK(TypedDict):
    # 지정가 + Fill-Or-Kill
    # 전량 즉시 체결되지 않으면 주문 취소
    quote_size: str
    base_size: str
    limit_price: str


class TwapLimitGTD(TypedDict):
    # TWAP (Time-Weighted Average Price) 알고리즘 주문 + 지정가 + GTD
    # 긴 시간 동안 거래량을 나눠서 평균 가격으로 매매
    quote_size: str
    base_size: str
    start_time: str      # TWAP 시작 시간 (ISO8601)
    end_time: str        # TWAP 종료 시간 (ISO8601)
    limit_price: str
    number_buckets: str  # 주문을 나눌 구간 수 (예: '10' → 10회로 분할)
    bucket_size: str     # 각 구간당 주문 수량
    bucket_duration: str # 각 구간 간격 시간 (초 단위)


class StopLimitStopLimitGTC(TypedDict):
    # 스탑 리밋 주문 (GTC)
    # 트리거 가격에 도달 시 지정가 주문 활성화
    base_size: str
    limit_price: str
    stop_price: str  # 트리거 가격 (이 가격 도달 시 지정가 주문 발생)
    stop_direction: Literal["STOP_DIRECTION_STOP_UP", "STOP_DIRECTION_STOP_DOWN"]
    # STOP_UP: 가격 상승 시 트리거 / STOP_DOWN: 가격 하락 시 트리거


class StopLimitStopLimitGTD(TypedDict):
    # 스탑 리밋 주문 (GTD)
    # 지정한 시간까지 유효한 스탑 리밋 주문
    base_size: str
    limit_price: str
    stop_price: str
    end_time: str
    stop_direction: Literal["STOP_DIRECTION_STOP_UP", "STOP_DIRECTION_STOP_DOWN"]


class TriggerBracketGTC(TypedDict):
    # 트리거 브래킷 주문 (GTC)
    # 스탑 리밋과 유사하지만, 스탑 트리거 후 지정가 주문 발생
    base_size: str
    limit_price: str
    stop_trigger_price: str  # 트리거 발생 가격


class TriggerBracketGTD(TypedDict):
    # 트리거 브래킷 주문 (GTD)
    base_size: str
    limit_price: str
    stop_trigger_price: str
    end_time: str  # 트리거 만료 시간


# -----------------------------
# Coinbase OrderConfiguration and AttachedOrderConfiguration
# -----------------------------

class OrderConfiguration(TypedDict, total=False):
    # 주문 구성: 아래 중 **하나만 활성화** 가능 (Order Type에 따라 선택)
    market_market_ioc: Optional[MarketMarketIOC]
    sor_limit_ioc: Optional[SorLimitIOC]
    limit_limit_gtc: Optional[LimitLimitGTC]
    limit_limit_gtd: Optional[LimitLimitGTD]
    limit_limit_fok: Optional[LimitLimitFOK]
    twap_limit_gtd: Optional[TwapLimitGTD]
    stop_limit_stop_limit_gtc: Optional[StopLimitStopLimitGTC]
    stop_limit_stop_limit_gtd: Optional[StopLimitStopLimitGTD]
    trigger_bracket_gtc: Optional[TriggerBracketGTC]
    trigger_bracket_gtd: Optional[TriggerBracketGTD]


# -----------------------------
# Final Coinbase Spot Order Request
# -----------------------------

class CoinbaseSpotOrderRequest(TypedDict):
    client_order_id: str  # 유저 정의 주문 ID (UUID v4 필요, 중복 방지)
    product_id: str       # 거래쌍 (예: "BTC-USD")
    side: OrderSide       # 'buy' or 'sell'
    order_configuration: OrderConfiguration  # 주문 유형과 세부 조건

    leverage: Optional[str]  # 레버리지 비율 (Spot에서는 보통 사용 안 함)
    margin_type: Optional[Literal["CROSS", "ISOLATED"]]  
    # 마진 유형: CROSS(교차) or ISOLATED(격리), Spot은 보통 None

    retail_portfolio_id: Optional[str]  # 개인 포트폴리오 식별 ID (주로 기관 계정용)
    preview_id: Optional[str]           # 사전 주문 미리보기 ID (주문 시뮬레이션 결과 활용)
    attached_order_configuration: Optional[OrderConfiguration]  
    # OCO(One-Cancels-the-Other) 등 복합 주문에 사용되는 추가 주문 구성