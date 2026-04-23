FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
# Add a 1000-second timeout and retries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt
COPY . .
# Expose Streamlit port
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]