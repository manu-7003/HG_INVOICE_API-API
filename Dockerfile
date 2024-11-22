# Use a slim Python image
FROM python:3.11-slim

# Install system-level dependencies, including zbar
RUN apt-get update && apt-get install -y libzbar0

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render will use
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
