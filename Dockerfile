FROM python:3.11-slim

WORKDIR /app

# Install dependencies
# We install fastembed, fastapi, and uvicorn
RUN pip install --no-cache-dir fastembed fastapi uvicorn[standard]

# Copy the app code
COPY main.py .

# Expose the port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
