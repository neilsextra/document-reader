# Use an official Python runtime as a parent image
FROM nikolaik/python-nodejs:python3.7-nodejs13

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN npm install -g bower

RUN echo '{ "allow_root": true }' > /root/.bowerrc

RUN  bower install

# Make port 80 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME xray-analyzer

# Run app.py when the container launches
CMD ["python", "app.py"]