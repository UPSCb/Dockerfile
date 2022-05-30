FROM ubuntu:20.04
MAINTAINER Dominik Stroh (stroh@stud.uni-frankfurt.de)

RUN apt-get update \
&& DEBIAN_FRONTEND=noninteractive \
apt install -y build-essential  unzip wget

RUN	mkdir build
RUN mkdir gemma
WORKDIR /build

RUN   wget https://github.com/genetics-statistics/GEMMA/releases/download/v0.98.5/gemma-0.98.5-linux-static-AMD64.gz \
&&    gzip -d gemma-0.98.5-linux-static-AMD64.gz \
&&    mv gemma-0.98.5-linux-static-AMD64 /usr/bin/gemma \
&&    chmod a+x /usr/bin/gemma

WORKDIR /
RUN rm -rf build

CMD ["gemma","-h"]
