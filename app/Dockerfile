# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file from the app directory
COPY ./app/requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the entire app directory into the container
COPY ./app /code/app

# Set environmental variable to ensure Streamlit runs on port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the app from its subdirectory
CMD ["streamlit", "run", "app/app.py"]
