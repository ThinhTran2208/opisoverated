FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASHION_INFERENCE_DEVICE=cpu

WORKDIR /app

COPY requirements-runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.inference.http_api:app", "--host", "0.0.0.0", "--port", "8000"]
