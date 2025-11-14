FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create frames directory
RUN mkdir -p /app/frames

# Copy project files
COPY pyproject.toml ./
COPY app ./app

# Install dependencies with CPU-only PyTorch and TensorFlow
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir tensorflow-cpu tensorflow-hub && \
    pip install --no-cache-dir -e . --no-deps && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pydantic-settings \
    sqlalchemy psycopg2-binary boto3 alembic python-multipart \
    pillow numpy av opencv-python-headless pandas openai



EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]