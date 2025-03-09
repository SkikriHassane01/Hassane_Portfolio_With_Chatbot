# Use multi-stage build to optimize the final image
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements
COPY requirements-prod.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements-prod.txt

# Create final image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=8000

# Copy wheels from builder stage
COPY --from=builder /app/wheels /wheels
# Install python packages without downloading again
RUN pip install --no-cache /wheels/*

# Copy only the necessary files for the application
COPY app.py config.py ./
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Create directories for logs
RUN mkdir -p logs

# Expose the port the app will run on
EXPOSE 8000

# Run the application using Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app