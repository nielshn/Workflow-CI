FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

COPY MLProject/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install mlflow==2.22.0

COPY MLProject/ .

CMD ["mlflow", "run", ".", "-e", "main", "--env-manager=local"]


