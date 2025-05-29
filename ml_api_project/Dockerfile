# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
# We need to install torch separately first for some environments
# or to ensure we get the CPU version if not otherwise specified by transformers.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY api.py .
# If you had other modules for the API, copy them too.

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
# Use uvicorn to run the FastAPI application
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]