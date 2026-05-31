# Hugging Face Spaces (Docker SDK) image for the Fitphone backend.
# Exposes the FastAPI app on the port HF expects (7860).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Build deps for some of RecBole's transitive scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user with uid 1000; matching avoids permission
# issues when the runtime writes the SQLite file or downloads cache files.
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python deps via pip from a requirements file derived from
# pyproject.toml — keeps the image small and avoids needing uv at runtime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API actually needs (skip notebooks, raw data, logs).
COPY src/ ./src/
COPY config/ ./config/
COPY assets/ ./assets/
COPY data/recbole/ ./data/recbole/
COPY saved/ ./saved/

# Spaces' container filesystem is ephemeral; the SQLite DB lives in /tmp
# so a restart just wipes demo events without surprising failures.
ENV DB_PATH=/tmp/events.db
RUN chown -R user:user /app

USER user
EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
