FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir -e /app

ENV PORT=8080
CMD ["uvicorn", "healthcare_ml.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
