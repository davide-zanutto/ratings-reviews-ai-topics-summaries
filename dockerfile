# Use an official Python slim image (change to a CUDA base if you need GPU support)
FROM python:3.12-slim

# 1. Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set workdir
WORKDIR /app

# 3. Copy requirements (you can also pip‑install directly in the next step)
COPY fine-tuning-scripts/azure_requirements.txt .

# 4. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r azure_requirements.txt

# 5. Copy your training code into the container
COPY fine-tuning-scripts/ ./fine-tuning-scripts/


CMD ["python", "fine-tuning-scripts/deBERTa-wandb.py"]
