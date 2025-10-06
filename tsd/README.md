# TSD-cpp251006

<<<<<<< HEAD
# Trading System Domain (TSD) Integration Guide

This README captures the code-facing work required to connect the existing Trading System Domain (TSD) stack with the multi-agent orchestration service so the combined system can function as an agent-assisted AI trader.

## 1. Grounding: What Already Works
- **Market plumbing (DSM)**: WebSocket/ZMQ publishers and parsing utilities provide real-time order book streaming primitives (`tsd/dsm`).
- **Order execution shell (OSM)**: Authenticated order managers, latency metrics, and signature helpers exist for Binance, Bybit, Coinbase, and OKX (`tsd/osm`).
- **Type system & config scaffolding**: Shared enums/structs for instruments, orders, and routing plus environment-driven config loaders are in place (`tsd/core`, `tsd/config`).
- **Multi-agent host**: The FastAPI service can orchestrate tool-using Gemini agents and already handles asynchronous message lifecycles (`backend/back-end`).

## 2. Key Gaps to Close
- **Integration surface**: No adapters expose DSM/OSM/TSM functionality as callable tools for the multi-agent backend.
- **Configuration mismatch**: `OrderServiceManager` expects lowercase keys, while `config.EXCHANGE_CONFIG` uses enum keys and omits REST URLs.
- **Strategy layer**: `tsd/tsm` lacks executable strategies, signals, or evaluation loops.
- **Knowledge persistence**: Streaming data is not persisted for later retrieval/analysis by agents; no external intelligence store exists.
- **Agent prompts & tooling**: Current Gemini prompts focus on CAE workflows, not trading-specific reasoning, risk, or execution.

## 3. Code-Level Action Plan
1. **Normalize Order Service configuration**
   - Extend `config.EXCHANGE_CONFIG` with REST/WebSocket endpoints keyed by lowercase exchange names or refactor `OrderServiceManager` to accept `Exchange` enums directly.
   - Audit each order manager constructor (`tsd/osm/managers/*.py`) to ensure consistent parameter order and error handling.

2. **Expose TSD primitives as agent tools**
   - Implement a thin service in `backend/back-end/node` (or a new `backend/back-end/trading_tools`) that:
     - Wraps DSM data access (e.g., `get_order_book`, `stream_quotes`).
     - Invokes OSM order managers through a safe execution facade (position/risk checks before order dispatch).
     - Returns structured payloads ready for Gemini function calls.
   - Register these functions with the host agent via tool definitions so the agent can plan → execute trading actions.

3. **Stand up persistence/knowledge store**
   - Choose TimescaleDB/Influx (matching existing docs) and add a recorder in `tsd/dsm/recorder` to persist normalized order books/trades/news.
   - Provide query utilities (e.g., SQL helpers or ORM) that agents can call to fetch historical context.

4. **Implement the agentic intelligence crawler**
   - Create an ingestion worker (`agentic_crawler` service) that scrapes APIs/RSS/social feeds, enriches the data, and stores it alongside market data.
   - Publish summaries or embeddings the multi-agent backend can read (consider a vector store if semantic retrieval is needed).

5. **Develop the Trading Strategy Manager (TSM)**
   - Prototype baseline strategies (e.g., mean reversion, momentum) inside `tsd/tsm/training` or a new `tsd/tsm/strategies` package.
   - Define interfaces for signal generation, backtesting, and live deployment; surface status/results through agent tools.

6. **Refactor agent prompts & orchestration for trading**
   - Replace the CAE-focused prompt instructions with trading roles (market analyst, risk manager, execution agent).
   - Keep the host-agent hierarchy but ensure outputs map to the new toolset and enforce risk guardrails in responses.

7. **Testing & validation**
   - Add integration tests covering order submission flows using sandbox endpoints (mock HTTP via `httpx.MockTransport`).
   - Write unit tests for new tool wrappers and config helpers to prevent regression when switching to local LLMs later.

## 4. Future Considerations
- **Local LLM support**: Abstract the agent runner so Gemini can be swapped for a local model with minimal code changes (e.g., dependency inversion for the `Runner`).
- **Observability**: Instrument DSM/OSM with metrics (Prometheus/OpenTelemetry) to monitor latency and agent decision loops.
- **Risk & compliance**: Plan for position limits, PnL tracking, and audit trails before enabling live trading.

Use this plan as the authoritative checklist when implementing the AI trader integration.
=======
# TSD-cpp251006
>>>>>>> tsdcppremote
