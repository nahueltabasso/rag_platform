FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=nointeractive

ENV PYTHONUNBUFFERED=1
ENV DIR=/opt/project/crawler
ENV PYTHONPATH=/opt/project

RUN mkdir -p ${DIR}
WORKDIR ${DIR}

RUN apt-get update && \
    apt-get install --no-install-recommends -y git wget curl build-essential libpq-dev \
        python3-dev python3-pip python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . ${DIR}

RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["tail", "-f", "/dev/null"]
