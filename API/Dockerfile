# The base image
FROM python:3.8.10 

# Main working dir for subsequent commands
WORKDIR /app

# Copy the file containing the necessary python libraries and install them to the image
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the saved model and the web apps
COPY api /app/
COPY models /app/

EXPOSE 8000

# Run the server 
CMD ["uvicorn", "basic-app:app", "--host", "0.0.0.0", "--port", "8000"]