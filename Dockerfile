# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install runtime dependencies for ACV, Door, and SHM pipelines
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    openpyxl \
    scikit-learn \
    scipy

# Copy application code and model configurations
COPY . .

# Expose Cloud Run default port
EXPOSE 8080

# Launch server
CMD ["python3", "app.py"]
