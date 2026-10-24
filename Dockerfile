FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY main.py .

# Default command runs the API; the worker service overrides it in docker-compose.yml
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]