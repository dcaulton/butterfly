FROM python:3.8.13 as builder
EXPOSE 8002
WORKDIR /app 
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /app 
ENV DJANGO_SETTINGS_MODULE=server.settings
