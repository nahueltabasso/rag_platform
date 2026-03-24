FROM python:3.12-slim-bullseye

ARG DEBIAN_FRONTEND=nointeractive

ENV PYTHONUNBUFFERED=1
ENV DIR=/opt/project
ENV PYTHONPATH=/opt/project

RUN mkdir -p ${DIR} /opt/models
WORKDIR ${DIR}

RUN apt-get update

