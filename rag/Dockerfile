FROM python:3.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV DIR=/opt/project/rag
ENV PYTHONPATH=/opt/project

RUN mkdir -p ${DIR}
WORKDIR ${DIR}
COPY . ${DIR}

RUN apt-get update && \
    apt-get install --no-install-recommends -y git wget vim iputils-ping curl build-essential libpq-dev \
        python3-dev python3-pip python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--", "--config_path", "config/wwII_wiki_config.json"]

