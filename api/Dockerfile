FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3.7 python3-distutils curl sudo git mecab mecab-ipadic-utf8 libmecab-dev


WORKDIR /downloads
RUN curl https://bootstrap.pypa.io/get-pip.py get-pip.py | python3.7

RUN pip install Flask &&\
    pip install flask-cors &&\
    pip install scikit-learn &&\
    pip install pandas &&\
    pip install mecab-python3 

COPY src/ /home/src

WORKDIR /home/src

EXPOSE 5000