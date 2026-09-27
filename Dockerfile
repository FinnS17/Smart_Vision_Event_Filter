FROM python:3.14-slim-bookworm

WORKDIR /app 

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN python -m pip install --no-cache-dir ".[ai]"
ENV SVF_DEVICE=cpu

ENTRYPOINT ["python", "-m", "smart_vision_event_filter"]