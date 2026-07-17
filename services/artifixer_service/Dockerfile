FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFIXER_DATA_ROOT=/data/jobs \
    ARTIFIXER_PIPELINE=stub \
    ARTIFIXER_CHECKPOINT=/models/artifixer/artifixer-14b.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY services/artifixer_service/requirements.txt /app/services/artifixer_service/requirements.txt
RUN pip3 install --no-cache-dir -r /app/services/artifixer_service/requirements.txt

# WorldAct contracts + service code (read-only runtime copy).
COPY worldact /app/worldact
COPY services /app/services

VOLUME ["/data/jobs", "/models/artifixer"]

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "services.artifixer_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
