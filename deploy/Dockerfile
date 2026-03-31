# Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# (Make sure you ran `pip freeze > requirements.txt` recently!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the standard OpenEnv port
EXPOSE 8000

# The command that Hugging Face Spaces will use to start your environment
CMD ["uvicorn", "env_server.main:app", "--host", "0.0.0.0", "--port", "8000"]