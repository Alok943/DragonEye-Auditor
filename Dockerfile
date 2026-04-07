FROM python:3.11-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy the project files
COPY . .

# Install dependencies
RUN uv pip install --system .

EXPOSE 7860

# Set Python path so it recognizes the src directory
ENV PYTHONPATH=/app/src

# Point Uvicorn to the server app inside the src layout
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]