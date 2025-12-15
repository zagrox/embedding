FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies required for AI/ONNX
# libgomp1 is CRITICAL for FastEmbed to work
RUN apt-get update && apt-get install -y \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
RUN pip install --no-cache-dir fastembed fastapi uvicorn[standard]

# 3. Create a writable cache directory for the model
# This ensures we don't get permission errors
RUN mkdir -p /app/fastembed_cache && chmod 777 /app/fastembed_cache

# Copy code
COPY main.py .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
