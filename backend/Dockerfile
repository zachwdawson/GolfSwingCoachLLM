FROM python:3.11

# Install system libraries for ffmpeg and Postgres client
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libpq-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create frames directory
RUN mkdir -p /app/frames

# Copy project files
COPY pyproject.toml ./
COPY app ./app

# Install dependencies and package
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

EXPOSE 8000

# ... existing content ...

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]
