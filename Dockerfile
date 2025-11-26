# Use a Python base image, choosing a stable version for consistency
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for Streamlit and Python
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV PYTHONUNBUFFERED 1

# Install system dependencies (needed for compilation/libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (leverages Docker cache)
COPY requirements.txt .

# Create the virtual environment and install packages (the heaviest step)
RUN python -m venv venv && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py, model files, etc.)
COPY . .

# Expose the port used by Streamlit
EXPOSE 8080

# Define the command to run when the container starts
CMD ["./venv/bin/streamlit", "run", "app.py"]
