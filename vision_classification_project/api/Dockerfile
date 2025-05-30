# api/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the API requirements file
COPY requirements_api.txt .

# Install API dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the API application code (main.py, preprocessing.py)
COPY main.py .
COPY preprocessing.py .

# --- MODEL COPY SECTION ---
# Create a directory inside /app for the model
RUN mkdir -p /app/model_weights
# Copy the model from the host (relative to this Dockerfile's location, which is api/)
# into /app/model_weights/ inside the container.
COPY ../models/cats_dogs_best_model.pth /app/model_weights/cats_dogs_best_model.pth
# IMPORTANT: If your model file has a DIFFERENT NAME, change it here.
# For example, if it's "my_model.pth":
# COPY ../models/my_model.pth /app/model_weights/my_model.pth

# Set environment variable for main.py to know it's in Docker
ENV RUNNING_IN_DOCKER=true

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]