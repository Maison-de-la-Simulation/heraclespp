# SPDX-License-Identifier: MIT

FROM ubuntu:jammy

ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]


RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && wget -O /usr/share/keyrings/pdidev-archive-keyring.gpg https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu/pdidev-archive-keyring.gpg \
 && wget -O /usr/share/keyrings/cuda-archive-keyring.asc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
 && wget -O /usr/share/keyrings/rocm-archive-keyring.asc https://repo.radeon.com/rocm/rocm.gpg.key \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pdidev-archive-keyring.gpg] https://raw.githubusercontent.com/pdidev/repo/pdi-master.ubuntu jammy main" > /etc/apt/sources.list.d/pdi.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cuda-archive-keyring.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.asc] https://repo.radeon.com/rocm/apt/latest jammy main" > /etc/apt/sources.list.d/rocm.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm-archive-keyring.asc] https://repo.radeon.com/amdgpu/latest/ubuntu jammy main" > /etc/apt/sources.list.d/amdgpu.list \
 && wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
 && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    clang-format \
    clang-tidy \
    cmake \
    doxygen \
    git \
    openssh-client \
    pkg-config \
    python3-pip \
 && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8 \
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
    rocm-hip-sdk \
 && apt-get purge -y \
    apt-transport-https \
    apt-utils \
    wget \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /ci -m -U ci \
 && mkdir /data \
 && chown ci:ci /ci /data \
 && python3 -m pip install -U --no-cache-dir \
    h5py \
    matplotlib \
    numpy \
    pip \
    scipy

USER ci:ci
WORKDIR /ci
ENV HOME=/ci
ENV PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CMD ["/bin/bash", "-li"]
