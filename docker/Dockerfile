# SPDX-License-Identifier: MIT

FROM ubuntu:jammy

ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]


RUN export DEBIAN_FRONTEND=noninteractive \
 && echo tzdata tzdata/Areas string Etc | debconf-set-selections \
 && echo tzdata tzdata/Zones/Etc string UTC | debconf-set-selections \
 && apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && echo "deb [ arch=amd64 ] https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu jammy main" > /etc/apt/sources.list.d/pdi.list \
 && wget -O /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu/pdidev-archive-keyring.gpg \
 && chmod a+r /etc/apt/trusted.gpg.d/pdidev-archive-keyring.gpg \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    pkg-config \
    cmake \
    openssh-client \
    clang-format \
    doxygen \
    python3-pip \
 && apt-get install -y --no-install-recommends \
    pdidev-archive-keyring \
    libopenmpi-dev \
    libpdi-dev \
    pdiplugin-decl-hdf5-openmpi \
    pdiplugin-mpi-openmpi \
    pdiplugin-pycall \
    pdiplugin-set-value \
    pdiplugin-trace \
    pdiplugin-user-code \
 && apt-get purge -y \
    apt-transport-https \
    apt-utils \
    wget \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && python3 -m pip install -U --no-cache-dir \
    h5py \
    matplotlib \
    numpy \
    pip \
    scipy \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /ci -m -U ci \
 && mkdir /data \
 && chown ci:ci /ci /data

USER ci:ci
WORKDIR /ci
ENV HOME=/ci

CMD ["/bin/bash", "-li"]
