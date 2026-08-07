# Dockerfile — MalnutriSense API container
FROM python:3.11-slim
 
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*
 
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy application code
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY reports/tables/ reports/tables/
 
# Expose API port
EXPOSE 8000
 
# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
 
# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
