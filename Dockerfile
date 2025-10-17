FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i en_US en_US.UTF-8
RUN apt-get install -y vim less

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN mkdir -p /code
COPY ./Python/requirements.txt /code
WORKDIR /code

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt