FROM python:3.11-slim

# Install system dependencies for PyAudio, SpeechRecognition, and pydub
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    libasound-dev \
    libav-tools \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Fly
EXPOSE 8080

# Run the app with gunicorn (for production)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "App:app"]
