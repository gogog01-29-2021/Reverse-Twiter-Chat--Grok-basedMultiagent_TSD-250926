# multi-agent-cae-v0.2

## Running the stack

    docker compose up -d --build

Equivalent manual sequence:

    docker compose build
    docker compose up -d

Check status and logs:

    docker compose ps
    docker compose logs -f

## Choosing the agent model provider

Set the following environment variables in `backend/.env` to control which LLM backs the orchestrator:

| Variable | Description | Default |
| --- | --- | --- |
| `AGENT_PROVIDER` | `api` to use Google Gemini via `google-adk`, `local` to route requests to a self-hosted model | `api` |
| `LOCAL_AGENT_ENDPOINT` | HTTP endpoint the local model server exposes (expects JSON payload `{user_id, session_id, prompt}` and returns a `text` field) | `http://localhost:8001/generate` |
| `LOCAL_AGENT_TIMEOUT` | Request timeout (seconds) when talking to the local endpoint | `60` |

When `AGENT_PROVIDER=api`, the existing Gemini toolchain is instantiated. When `AGENT_PROVIDER=local`, the backend sends prompts to the configured endpoint, enabling you to plug in locally served models such as `gpt-oss` or `nanoGPT` with the same conversation workflow.

Install backend dependencies after switching providers:

    uv sync   # or your preferred pip/uv/poetry install command
