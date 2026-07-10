# Vive Health Quality Suite — Hugging Face Spaces (Docker SDK)
# Deterministic build honoring the exact pins in requirements.txt.
FROM python:3.12-slim

# System packages (mirrors packages.txt)
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user; ~/.quality_app must be writable (corrections memory,
# connection configs). Note: container storage is ephemeral across restarts.
RUN useradd -m appuser && \
    mkdir -p /home/appuser/.quality_app && \
    chown -R appuser:appuser /home/appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
