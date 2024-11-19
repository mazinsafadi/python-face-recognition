# Use TensorFlow base image
FROM tensorflow/tensorflow:2.13.0

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DeepFace model weights during the build phase
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

# Optimize TensorFlow for CPU (optional but recommended for performance)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose application port
EXPOSE 80

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
