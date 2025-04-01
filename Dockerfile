FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the application
COPY . .

# Create directory for output
RUN mkdir -p /app/output

# Create a non-root user
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python3", "main.py", "--simple-mode", "--optimize-for", "balanced", "--iterations", "1"] 