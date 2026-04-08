FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip --quiet && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY src/features.py ./src/features.py
COPY src/__init__.py ./src/__init__.py
COPY models/ ./models/

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]