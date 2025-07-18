FROM python:3.10-slim

# Install system dependencies required for PyAudio and ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files to container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "App:app"]
