FROM python:3.11-slim

WORKDIR /app

# Ensure we use UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The environment must listen on port 8000 for the automated evaluator
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]