FROM python:3.12-slim-bullseye

ARG DEBIAN_FRONTEND=nointeractive

ENV PYTHONUNBUFFERED=1
ENV DIR=/opt/project
ENV PYTHONPATH=/opt/project

RUN mkdir -p ${DIR} /opt/models
WORKDIR ${DIR}

RUN apt-get update  && \
    apt-get install --no-install-recommends -y git vim wget curl iputils-ping build-essential libpq-dev software-properties-common \
        python3-dev python3-pip python3-setuptools python3-distutils graphviz graphviz-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . ${DIR}

RUN pip3 install -r requirements.txt --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

