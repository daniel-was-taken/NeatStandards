# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install system dependencies required for sentence-transformers and OpenCV
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
    
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Copy and make entrypoint executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Make port 8000 available to the world outside this container (for local dev)
EXPOSE 8000

# Run the entrypoint script when the container launches
ENTRYPOINT ["./entrypoint.sh"]

# Make port 8000 available to the world outside this container
# EXPOSE 8000

# Define environment variable
# ENV CHAINLIT_HOST=0.0.0.0
# ENV CHAINLIT_PORT=8000

# Run app.py when the container launches
# CMD ["chainlit", "run", "app.py"]
# CMD chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000} --headless
