FROM python:3.6
COPY pip.conf /pip.conf
ENV PIP_CONFIG_FILE=/pip.conf
ENV AWS_DEFAULT_REGION=us-east-1
ENV IA_ENV=$IA_ENV
RUN pip3 install --upgrade pip setuptools
COPY requirements.txt / 
RUN pip3 install -r requirements.txt
COPY src/ /src