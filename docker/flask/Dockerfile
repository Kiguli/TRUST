FROM python:3.12.2-bookworm

LABEL maintainer="Jamie Gardner"

WORKDIR /var/www/html

ARG WWWGROUP
ARG NODE_VERSION=20

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP="main.py"
ENV FLASK_ENV="production"
ENV FLASK_DEBUG=0
ENV PIP_ROOT_USER_ACTION=ignore
ENV SUPERVISOR_FLASK_COMMAND="gunicorn -w 5 'main:create_app()' -b 0.0.0.0:8000 --timeout 1800 --access-logfile - --error-logfile -"
ENV SUPERVISOR_FLASK_USER="trustapp"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN apt-get update && apt-get upgrade -y \
    && mkdir -p /etc/apt/keyrings \
    && apt-get install -y gnupg gosu curl ca-certificates zip unzip git supervisor sqlite3 libcap2-bin libpng-dev dnsutils librsvg2-bin fswatch ffmpeg nano \
       build-essential libssl-dev libffi-dev libopenblas-dev

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm \
    && apt-get -y autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd --force -g $WWWGROUP trustapp
RUN useradd -ms /bin/bash --no-user-group -g $WWWGROUP -u 1337 trustapp

RUN setcap 'cap_net_bind_service=+ep' /usr/local/bin/python3.12

COPY docker/flask/start-container /usr/local/bin/start-container
COPY docker/flask/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chmod +x /usr/local/bin/start-container

COPY . .

RUN mkdir -p /var/www/html/storage/uploads \
    && chown -R trustapp:trustapp /var/www/html \
    && chmod -R 775 /var/www/html/storage

RUN pip install --upgrade pip \
    && pip install gunicorn \
    && pip install -r requirements.txt

RUN test -f .env || cp .env.example .env

RUN cd /var/www/html/vite \
    && npm ci \
    && npm run build

ENTRYPOINT ["start-container"]
