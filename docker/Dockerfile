FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates git wget sudo python3-dev python3-pip bzip2 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python && \
    # ln -sv /usr/bin/pip3 /usr/bin/pip && \ 
    pip3 install --upgrade pip 

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /work
RUN sudo chmod -R 777 /work
ENV PYTHONPATH='/work/:$PYTHONPATH'

COPY requirements.txt . 

RUN pip3 install --user -r requirements.txt --no-deps

CMD ["bash"]
