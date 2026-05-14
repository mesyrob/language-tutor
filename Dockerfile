FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy app code.
COPY main.py ./
COPY curriculum/ ./curriculum/

# Install the project itself so `uv run` resolves the entrypoint.
RUN uv sync --frozen --no-dev

# Persistent SQLite lives outside the image — mount a PVC at /data.
ENV DB_PATH=/data/tutor.db
VOLUME ["/data"]

# Drop privileges.
RUN useradd --uid 1000 --create-home tutor && \
    mkdir -p /data && chown -R tutor:tutor /data /app
USER tutor

CMD ["uv", "run", "--frozen", "--no-dev", "main.py"]
