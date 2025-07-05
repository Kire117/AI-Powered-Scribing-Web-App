# Use Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port (Flask default: 5000, FastAPI: 8000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]