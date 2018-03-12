FROM delhomme/upscb-jbrowse
MAINTAINER Niklas Mähler (niklas.mahler@gmail.com)

RUN mkdir build
WORKDIR /build

RUN apt update && apt-get install -y \
    default-jre \
    git \
    less \
    libbz2-dev \
    liblzma-dev \
    man \
    nano \
    python3-dev \
    python-pip \
    openssh-server \
    openssl \
    pkg-config \
    vim \
    wget

# BWA
RUN wget https://sourceforge.net/projects/bio-bwa/files/bwa-0.7.17.tar.bz2 && \
    tar jxf bwa-0.7.17.tar.bz2 && \
    cd bwa-0.7.17 && make && \
    cp bwa /usr/local/bin

# Samtools
WORKDIR /
RUN wget https://github.com/samtools/samtools/releases/download/1.7/samtools-1.7.tar.bz2 && \
    tar jxf samtools-1.7.tar.bz2 && \
    cd samtools-1.7 && \
    ./configure && \
    make && \
    make install

# GATK
RUN mkdir /tools
RUN wget https://github.com/broadinstitute/gatk/releases/download/4.0.1.2/gatk-4.0.1.2.zip && \
    unzip gatk-4.0.1.2.zip && \
    cp -r gatk-4.0.1.2 /tools && \
    ln -s /tools/gatk-4.0.1.2/gatk /usr/local/bin

# Picard
WORKDIR /build
RUN wget https://github.com/broadinstitute/picard/releases/download/2.17.10/picard.jar && \
    cp picard.jar /tools && \
    echo "alias picard='java -jar /tools/picard.jar'" >> /etc/profile.d/00-aliases.sh

# freebayes
RUN git clone --recursive git://github.com/ekg/freebayes.git && \
    cd freebayes && \
    make && \
    make install

# Qualimap
WORKDIR /build
RUN wget https://bitbucket.org/kokonech/qualimap/downloads/qualimap_v2.2.1.zip && \
    unzip qualimap_v2.2.1.zip && \
    mv qualimap_v2.2.1 /tools && \
    ln -s /tools/qualimap_v2.2.1/qualimap /usr/local/bin

# plink
RUN wget http://s3.amazonaws.com/plink2-assets/alpha1/plink2_linux_avx2.zip && \
    unzip plink2_linux_avx2.zip && \
    cp plink2 /usr/local/bin && \
    echo "alias plink=plink2" >> /etc/profile.d/00-aliases.sh

# vcftools
RUN wget https://github.com/vcftools/vcftools/releases/download/v0.1.15/vcftools-0.1.15.tar.gz && \
    tar xzvf vcftools-0.1.15.tar.gz && \
    cd vcftools-0.1.15 && \
    ./configure && \
    make && \
    make install

# GateOne
RUN pip install --upgrade pip && pip install tornado==4.4.3 && \
    git clone https://github.com/liftoff/GateOne/ && \
    cd GateOne && python setup.py install && \
    python run_gateone.py --configure && cd ..

# More aliasesa
RUN echo "alias ll='ls -lh'" >> /etc/profile.d/00-aliases.sh

# JBrowse stuff
RUN mkdir /var/www/html/JBrowse-1.12.4/datafiles && \
    chown www-data:training /var/www/html/JBrowse-1.12.4/datafiles && \
    chmod 775 /var/www/html/JBrowse-1.12.4/datafiles

ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
VOLUME /var/www/html/JBrowse-1.12.1/data
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

RUN mkdir /home/training/share
VOLUME /home/training/share

WORKDIR /home/training
RUN rm -rf /build

EXPOSE 22 80 443
