# SPDX-License-Identifier: MIT

FROM ubuntu:noble

ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED 1
ENV PATH=/opt/nova-venv/bin:${PATH:+:${PATH}}
ENV PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get --yes update \
 && apt-get --yes upgrade \
 && apt-get --yes --no-install-recommends install \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && wget --output-document=/usr/share/keyrings/pdidev-archive-keyring.gpg https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu/pdidev-archive-keyring.gpg \
 && wget --output-document=/usr/share/keyrings/cuda-archive-keyring.asc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
 && wget --output-document=/usr/share/keyrings/rocm-archive-keyring.asc https://repo.radeon.com/rocm/rocm.gpg.key \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pdidev-archive-keyring.gpg] https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu noble main" > /etc/apt/sources.list.d/pdi.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cuda-archive-keyring.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.asc] https://repo.radeon.com/rocm/apt/6.2 noble main" > /etc/apt/sources.list.d/rocm.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.asc] https://repo.radeon.com/amdgpu/6.2/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list \
 && wget --output-document=/etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin \
 && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 && apt-get --yes update \
 && apt-get --yes --no-install-recommends install \
    build-essential \
    clang-format \
    clang-tidy \
    cmake \
    doxygen \
    git \
    openssh-client \
    pkg-config \
    python3-pip \
    python3-venv \
 && apt-get --yes --no-install-recommends install \
    cuda-minimal-build-12-6 \
    libhdf5-openmpi-dev \
    libopenmpi-dev \
    libpdi-dev \
    ninja-build \
    pdidev-archive-keyring \
    pdiplugin-decl-hdf5-openmpi \
    pdiplugin-mpi-openmpi \
    pdiplugin-pycall \
    pdiplugin-set-value \
    pdiplugin-trace \
    pdiplugin-user-code \
    rocm-hip-runtime-dev \
 && echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
 && echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf \
 && ldconfig \
 && python3 -m venv /opt/nova-venv \
 && . /opt/nova-venv/bin/activate \
 && python3 -m pip install --no-cache-dir --require-virtualenv --upgrade \
    h5py \
    matplotlib \
    numpy \
    pip \
    pyyaml \
    scipy \
 && deactivate \
 && apt-get --yes purge \
    apt-transport-https \
    apt-utils \
    wget \
 && apt-get --yes autoremove \
 && apt-get --yes clean \
 && apt-get --yes autoclean \
 && rm --force --recursive /var/lib/apt/lists/* \
 && useradd --home-dir /ci --create-home --user-group ci \
 && mkdir /data \
 && chown ci:ci /ci /data

USER ci:ci
WORKDIR /ci
ENV HOME=/ci

CMD ["/bin/bash", "-li"]
