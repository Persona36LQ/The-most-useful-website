FROM python:3.10 AS builder-django

WORKDIR /app/server
COPY ./server .

# I know about requirements.txt but it doesnt work no clue why
RUN pip install django djangorestframework django-cors-headers torch numpy opencv-python