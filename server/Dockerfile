FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3.7 python3-distutils curl


WORKDIR /downloads
RUN curl https://bootstrap.pypa.io/get-pip.py get-pip.py | python3.7

RUN pip install Flask 

COPY src/ /home/src
WORKDIR /home/src

EXPOSE 8080