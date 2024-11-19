FROM ubuntu:24.04

LABEL maintainer="Jamie Gardner"

ARG WWWGROUP
ARG NODE_VERSION=20

WORKDIR /var/www/html

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV FLASK_APP="main.py"
ENV FLASK_ENV="development"
ENV FLASK_DEBUG=1

ENV SUPERVISOR_FLASK_COMMAND="flask run --host=0.0.0.0"
ENV SUPERVISOR_FLASK_USER="trustapp"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN apt-get update && apt-get upgrade -y \
    && mkdir -p /etc/apt/keyrings \
    && apt-get install -y gnupg gosu curl ca-certificates zip unzip git supervisor sqlite3 libcap2-bin libpng-dev dnsutils librsvg2-bin fswatch ffmpeg nano \
       build-essential libssl-dev libffi-dev python3-pip python3-dev python3-setuptools python3-venv libopenblas-dev

RUN mkdir -m 0755 -p /etc/apt/keyrings/ \
    && curl -s https://deb.nodesource.com/gpgkey/nodesource.gpg.key | gpg --dearmor | tee /etc/apt/keyrings/nodesource.gpg >/dev/null \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x jammy main" > /etc/apt/sources.list.d/nodesource.list \
    && echo "deb-src [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x jammy main" >> /etc/apt/sources.list.d/nodesource.list \
#    && echo "deb [signed-by=/etc/apt/nodesource.gpg] https://deb.nodesource.com/node_$NODE_VERSION.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && chmod a+r /etc/apt/keyrings/nodesource.gpg \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd --force -g $WWWGROUP trustapp
RUN useradd -ms /bin/bash --no-user-group -g $WWWGROUP -u 1337 trustapp

COPY docker/flask/start-container /usr/local/bin/start-container
COPY docker/flask/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chmod +x /usr/local/bin/start-container

ENV VIRTUAL_ENV=/var/www/html/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
COPY vite/package.json .
RUN . venv/bin/activate \
    && python -m ensurepip --default-pip \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \

ENTRYPOINT ["start-container"]
