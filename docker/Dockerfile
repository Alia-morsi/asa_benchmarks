#FROM continuumio/miniconda3
FROM       ubuntu:18.04 as builder
MAINTAINER Alia Morsi  "https://registry.sb.upf.edu"
LABEL authors="Alia Morsi"
LABEL version="18.04"
LABEL description="Image for Running DTW Audio to Score Alignment Experiments"
RUN apt-get update
RUN apt-get install -y openssh-server nmap sudo telnet sssd
RUN mkdir /var/run/sshd
RUN echo 'root:xxxxxxxxxxxx' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 22
CMD    ["/usr/sbin/sshd", "-D"]

FROM python:3.8.0 as app
#RUN sh install_fmpeg.sh
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y fluidsynth
RUN apt-get update && apt-get upgrade -y && apt-get install -y && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
RUN apt-get install -y vim
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install --default-timeout=10000 -r requirements.txt
RUN pip install torch torchvision
RUN pip install music21
RUN pip install -U llvmlite==0.32.1
RUN pip install ipython
RUN pip install madmom
RUN mkdir -p /root/vamp
RUN mkdir -p /root/asa_benchmarks
COPY vamp/* /root/vamp/
RUN pip install vamp
RUN pip install numba==0.53
COPY asa_benchmarks/ /root/asa_benchmarks/


#CMD echo 'hello world' 
