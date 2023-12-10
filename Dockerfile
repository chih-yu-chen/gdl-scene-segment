# syntax=docker/dockerfile:1

FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive; \
    apt update; \
    apt install --no-install-recommends -y \
        ca-certificates \
        build-essential \
        git \
        qt5-default; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD git clone https://github.com/JonasSchult/vcglib; \
    cd vcglib/apps/tridecimator/; \
    qmake; \
    make; \
    \
    cd ../sample/trimesh_clustering; \
    qmake; \
    make