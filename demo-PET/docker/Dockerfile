FROM ubuntu:jammy
#SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list
RUN apt update
RUN apt install -y \
  python3-pip \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libzmq5-dev python3-dev \
  build-essential swig cmake git

RUN pip install helics
## Java support
#RUN apt -y install openjdk-11-jdk
#RUN ln -sf /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java
#
## for HELICS and FNCS
#RUN apt -y install libzmq5-dev libczmq-dev libboost-dev
#
## for GridLAB-D
#RUN apt -y install libxerces-c-dev libhdf5-serial-dev
#
## for solvers used by AMES/Agents/GridLAB-D
## needed for KLU
#RUN apt -y install libsuitesparse-dev
#
## end users replace libsuitesparse-dev with libklu1, which is licensed LGPL
## needed for Ipopt/cbc
#RUN apt -y install coinor-cbc coinor-libcbc-dev coinor-libipopt-dev liblapack-dev libmetis-dev
#
#RUN apt -y install software-properties-common
#RUN add-apt-repository -y ppa:deadsnakes/ppa
#RUN apt -y install python3.8 python3.8-venv python3-pip python3.8-tk python3-pil.imagetk
#
#RUN apt -y install wget
#
#WORKDIR /root
#RUN mkdir -p tesp
#WORKDIR /root/tesp
#RUN python3.8 -m venv venv --prompt TESP
#
#RUN mkdir -p tenv
#RUN mkdir -p repository
#WORKDIR /root/tesp/repository
#RUN git clone -b v1.2.3 https://github.com/pnnl/tesp.git
#RUN echo "Copy TESP environment variables to $HOME/tespEnv for shell scripts"
#RUN cp tesp/scripts/tespEnv "$HOME/"
#RUN git clone -b master https://github.com/ames-market/AMES-V5.0.git
#RUN . "${HOME}/tespEnv" && "${TESPBUILD}/patch.sh" AMES-V5.0 AMES-V5.0
##RUN . "${HOME}/tespEnv" && cd "${TESPBUILD}" && ./tesp_c.sh $binaries
#RUN . "${HOME}/tespEnv" && pip3 install -r "${TESPDIR}/requirements.txt"
#RUN . "${HOME}/tespEnv" && cd "${TESPDIR}/src/tesp_support" && pip3 install -e .
#RUN . "${HOME}/tespEnv" && cd "${REPODIR}/AMES-V5.0/psst" && pip3 install -e .
#
#RUN . "${HOME}/tespEnv" && cd "${TESPDIR}/scripts" && ./tesp_c.sh copy
#
##ENV INSTDIR=$HOME/tesp/tenv
##ENV REPODIR=$HOME/tesp/repository
##ENV TESPDIR=$HOME/tesp/repository/tesp
##ENV TESPBUILD=$TESPDIR/scripts/build
##RUN . "${HOME}/tespEnv" && cd "${REPODIR}/AMES-V5.0/psst" && pip3 install -e . > "${TESPBUILD}/tesp_api.log" 2>&1
##WORKDIR $INSTDIR
##RUN wget --no-check-certificate https://github.com/pnnl/tesp/releases/download/v1.2.3/tesp_binaries.zip
##RUN unzip tesp_binaries.zip
##RUN rm tesp_binaries.zip
##
##WORKDIR $TESPBUILD
##RUN ls -l
##RUN ls ..
##RUN . "${HOME}/tespEnv" && ./HELICS-py.sh clean
##
##RUN ldconfig
##RUN $TESPBUILD/versions.sh
#
#
#
#
#
##RUN python3.8 -m pip install virtualenv
##RUN "${HOME}/.local/bin/virtualenv" venv --prompt TESP
#
##RUN wget --no-check-certificate https://raw.githubusercontent.com/pnnl/tesp/main/scripts/tesp.sh
##RUN sed -i 's/sudo //' tesp.sh
##RUN chmod 755 tesp.sh
##RUN yes | ./tesp.sh