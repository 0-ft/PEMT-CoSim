FROM ubuntu:jammy
SHELL ["/bin/bash", "-l", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG HOME=/root
RUN apt update
RUN apt install -y \
  python3-pip \
  git \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libzmq5-dev python3-dev \
  build-essential swig cmake git

RUN apt install -y nano libxtst6 libxi6 libxrender1 wget firefox

WORKDIR $HOME

# install GridLAB-D
RUN wget https://github.com/gridlab-d/gridlab-d/releases/download/v5.1/GridLAB-D-5.1.0-Linux.deb -q
RUN dpkg -i GridLAB-D-5.1.0-Linux.deb

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -q
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH /root/miniconda3/bin:$PATH
RUN sed -e '/[ -z "$PS1" ] && return/s/^/#/g' -i /root/.bashrc

# install HELICS
RUN wget https://github.com/GMLC-TDC/HELICS/releases/download/v3.5.2/Helics-3.5.2-Linux-x86_64.tar.gz -O helics.tar.gz -q
RUN mkdir helics
RUN tar -xzf helics.tar.gz -C helics --strip-components=1
RUN cp -r helics/* /usr/local
RUN ldconfig
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64" >> ~/.bashrc

WORKDIR $HOME

# create conda environment & install python requirements
RUN /root/miniconda3/bin/conda init bash
COPY conda_env.yml .
RUN conda env create -f conda_env.yml
RUN echo "conda activate cosim" >> ~/.bashrc
RUN emobpy create -n unused

WORKDIR /PEMT-CoSim

RUN git config --global --add safe.directory /PEMT-CoSim
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "cosim", "python3", "-u", "generate_case.py"]
