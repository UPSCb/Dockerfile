FROM ubuntu

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libz-dev \
        libbz2-dev \
        liblzma-dev \
        git

RUN git clone git://github.com/samtools/htslib.git && \
    git clone git://github.com/samtools/bcftools.git

WORKDIR bcftools

RUN make && make install
