FROM python:latest AS builder-django

WORKDIR /app/server
COPY ./server .

# I know about requirements.txt but it doesnt work no clue why

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#RUN apt-get -y update && apt -y install nano # If you need nano

RUN pip install django djangorestframework django-cors-headers torch numpy opencv-python



