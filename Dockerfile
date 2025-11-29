# Use Python 3.10
FROM python:3.10-slim

# 1. Force logs to show immediately
ENV PYTHONUNBUFFERED=1

# 2. Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*

# 3. Set up working directory
WORKDIR /app

# 4. Create necessary folders
RUN mkdir -p /app/rvc/models/predictors \
    && mkdir -p /app/rvc/models/embedders/contentvec \
    && mkdir -p /app/rvc/models/pretraineds/hifi-gan \
    && mkdir -p /app/uploads \
    && mkdir -p /app/logs \
    && mkdir -p /app/datasets \
    && mkdir -p /app/model

# 5. Copy requirements
COPY requirements.txt .

# 6. INSTALL GUNICORN EXPLICITLY (The Fix)
# We run this BEFORE requirements to ensure it exists
RUN pip install --no-cache-dir gunicorn

# 7. Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy application code
COPY . .

# 9. Grant permissions
RUN chmod -R 777 /app/rvc \
    && chmod -R 777 /app/uploads \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/datasets \
    && chmod -R 777 /app/model

# 10. Expose port
EXPOSE 7860

# 11. Run app (Reverted to standard command now that we know it will be installed)
# Increase timeout to 300 seconds (5 minutes)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "600"]