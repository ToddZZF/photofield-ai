# Runtime
FROM python:3.13-slim
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using system Python
RUN uv sync --frozen --no-dev --no-install-project --python $(which python3)

COPY models/*.onnx models/
COPY models/*.onnx.data models/
COPY cliponnx cliponnx
COPY main.py ./

EXPOSE 8081
CMD ["uv", "run", "--no-dev", "python", "main.py"]
