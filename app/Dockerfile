# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# FIX: The workflow uploads requirements.txt to the ROOT of the Space
# So we copy it from the current build context (.) to /code/
COPY requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# FIX: The workflow uploads app.py into an 'app/' folder
# We copy that folder into our container
COPY app /code/app

# Set environmental variables for Hugging Face
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the app
CMD ["streamlit", "run", "app/app.py"]
