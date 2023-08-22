#FROM --platform=$BUILDPLATFORM python:3.7-alpine AS builder
FROM python:3.8.13 as builder
#RUN apt-get -y update && apt-get -y install build-essential 
EXPOSE 8002
WORKDIR /app 
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
COPY . /app 
#ENTRYPOINT ["python3"] 
#CMD ["daphne", "-p", "8002", "server.asgi:application"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8002"]

#FROM builder as dev-envs
#RUN <<EOF
#apk update
#apk add git
#EOF
#
#RUN <<EOF
#addgroup -S docker
#adduser -S --shell /bin/bash --ingroup docker vscode
#EOF
## install Docker tools (cli, buildx, compose)
#COPY --from=gloursdocker/docker / /
#CMD ["manage.py", "runserver", "0.0.0.0:8000"]

#FROM python
#
#RUN mkdir /butterfly
#WORKDIR /butterfly
#COPY . /butterfly
#
#EXPOSE 8000
#RUN pip install -r requirements.txt
#CMD ["python", "manage.py", "runserver"]
