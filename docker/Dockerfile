# Simple API image (optional)
FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY api/ api/
COPY scripts/ scripts/
COPY schema/ schema/
COPY README.md .

# Install minimal deps; extend as needed
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

