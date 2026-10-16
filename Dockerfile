# Base image: official slim python
FROM python:3.12-slim

# Don't write .pyc files, print output immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only what pip needs first
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[dev]"

# Copy the rest
COPY . .

# Default: Run the test suite. Override at "docker run" step to do anything else