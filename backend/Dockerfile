FROM python:3.12-slim

WORKDIR /app

# Install dependencies directly with pip
COPY pyproject.toml ./
RUN pip install -e .

COPY . .

CMD ["uvicorn", "back-end.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]