# Base image
FROM python:3.11

# Install all required packages to run the model
RUN apt update && apt install --yes ffmpeg libsm6 libxext6
