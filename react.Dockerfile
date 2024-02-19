FROM node:14 AS builder-react

WORKDIR /app/client

COPY ./client/package*.json ./
RUN npm install
COPY client .
RUN npm run build