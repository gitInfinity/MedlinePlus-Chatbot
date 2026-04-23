FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
# Add a 1000-second timeout and retries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt
COPY backend/ .
COPY config/ ./config/
EXPOSE 8000
CMD ["uvicorn", "retrieve_api:app", "--host", "0.0.0.0", "--port", "8000"]