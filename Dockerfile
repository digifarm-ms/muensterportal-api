FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files and source code
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies and the package
RUN uv sync --frozen --no-cache --no-dev

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "--no-sync", "uvicorn", "muenster4you:app", "--host", "0.0.0.0", "--port", "8000"]