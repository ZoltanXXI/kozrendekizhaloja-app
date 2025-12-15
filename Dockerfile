# Dockerfile for running the Streamlit app
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install system deps for common wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app sources
COPY . /app

# Default Streamlit port
EXPOSE 8501

# Run Streamlit (Render / Docker platforms usually inject PORT env var)
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
