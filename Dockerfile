# Use Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose FastAPI port
EXPOSE 8001

# Expose Streamlit port
EXPOSE 8502

# Run FastAPI backend and Streamlit frontend
CMD ["sh", "-c", "uvicorn Main:app --host 0.0.0.0 --port 8001 & streamlit run App.py --server.address 0.0.0.0 --server.port 8502"]