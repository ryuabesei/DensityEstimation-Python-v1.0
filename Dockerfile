FROM python:3
USER root


ENV LANG=en_US.UTF-8\
    LANGUAGE=en_US:en\
    LC_ALL=en_US.UTF-8\
    TZ=JST-9 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TERM=xterm

RUN apt-get update && apt-get install -y --no-install-recommends\
    tzdata\
    build-essential\
    locales\
    gfortran\
    meson\
    ninja-build\
    pkg-config\
    libgeos-dev\
    libproj-dev\
    proj-data\
    proj-bin\
    vim\
    less\
    && localedef -f UTF-8 -i en_US en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./Python/requirements.txt /code


RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt