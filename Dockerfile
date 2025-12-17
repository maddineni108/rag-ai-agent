# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir is used to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# After pip install
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000 for the API
EXPOSE 8000

# Define environment variable to ensure python output is sent straight to terminal (unbuffered)
ENV PYTHONUNBUFFERED=1

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
