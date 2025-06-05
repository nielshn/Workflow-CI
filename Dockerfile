FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install MLflow (pastikan setelah dependensi sistem)
RUN pip install mlflow==2.22.0

# Copy source code
COPY . .

# Set env vars if needed
ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["python", "modelling.py"]
