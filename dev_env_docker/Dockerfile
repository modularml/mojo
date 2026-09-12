FROM ubuntu:latest

ENV PATH="/root/.pixi/bin:${PATH}"

RUN apt-get update

RUN apt-get install -y curl

RUN apt-get install -y git

RUN apt-get install -y llvm

RUN apt-get install -y python3 python3-pip

RUN curl -fsSL https://pixi.sh/install.sh | sh

RUN pixi init mojo-nightly-env -c https://conda.modular.com/max-nightly/ -c conda-forge

RUN cd mojo-nightly-env && pixi add modular

WORKDIR /mojo-nightly-env/mojo
