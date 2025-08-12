# Start with an official, lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port that Gunicorn will run on
EXPOSE 8050

# The command to run your application using the Gunicorn web server
# This is the production-ready way to run a Dash app.
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]