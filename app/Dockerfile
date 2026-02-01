# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory to /code
WORKDIR /code

# Copy requirements from the local path to the container
# If the Dockerfile is in root, use: COPY app/requirements.txt .
# If the Dockerfile is in app/, use: COPY requirements.txt .
COPY app/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files into the container
COPY . /code

# Standard Hugging Face port
EXPOSE 7860

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
