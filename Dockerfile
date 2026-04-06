FROM python:3.11-slim

WORKDIR /app

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY requirements.txt .

# Ensure pywin32 is REMOVED from your requirements.txt before this runs!
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces strictly listens on 7860
EXPOSE 7860

# THIS MUST BE THE ONLY CMD LINE IN THE ENTIRE FILE
CMD ["uvicorn", "env_server.main:app", "--host", "0.0.0.0", "--port", "7860"]