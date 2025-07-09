# Use a base image with Python
FROM python:3.10-slim

# Install ffmpeg and other dependencies
RUN apt-get update && apt-get install -y ffmpeg build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Run the app
CMD ["python", "App.py"]

