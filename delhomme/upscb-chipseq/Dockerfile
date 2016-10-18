FROM delhomme/upscb-ngs-preprocessing
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN	apt-get install -y libghc-zlib-dev libghc-zlib-dev libncurses-dev libbz2-dev \
liblzma-dev libpcre3-dev libxml2-dev libblas-dev gfortran git unzip ftp libzmq3-dev \
nano ftp fort77 libreadline-dev libcurl4-openssl-dev libx11-dev libxt-dev x11-common \
libcairo2-dev libpng12-dev libreadline6-dev libjpeg8-dev pkg-config
	
#########
### Working dir
#########
RUN	mkdir build 
WORKDIR /build

#########
### R version 3.3.1
#########
RUN wget http://ftp.acc.umu.se/mirror/CRAN/src/base/R-3/R-3.3.1.tar.gz && \
    tar -zxf R-3.3.1.tar.gz && cd R-3.3.1 && \
    ./configure --enable-memory-profiling -enable-R-shlib --with-blas --with-lapack \
	--with-valgrind-instrumentation=3 --with-readline --with-cairo --with-libpng --with-jpeglib --with-x=no && \
    make && make check && make pdf && make info && make install && make install-info && make install-pdf

#########
### HTSlib tools: Samtools, Bcftools
#########
RUN wget https://github.com/samtools/samtools/releases/download/1.3.1/samtools-1.3.1.tar.bz2 \
	https://github.com/samtools/bcftools/releases/download/1.3.1/bcftools-1.3.1.tar.bz2 \
	https://github.com/samtools/htslib/releases/download/1.3.1/htslib-1.3.1.tar.bz2 && \
	tar -xf htslib-1.3.1.tar.bz2
WORKDIR htslib-1.3.1
RUN ./configure && make && make install
WORKDIR /build
RUN tar -xf bcftools-1.3.1.tar.bz2
WORKDIR bcftools-1.3.1
RUN make && make install
WORKDIR /build
RUN tar -xf samtools-1.3.1.tar.bz2
WORKDIR samtools-1.3.1
RUN make && make install
WORKDIR /build

#########
### BEDtools
#########
RUN wget https://github.com/arq5x/bedtools2/releases/download/v2.26.0/bedtools-2.26.0.tar.gz && \
	tar -xf bedtools-2.26.0.tar.gz && cd bedtools2 && \
	make && make install

#########
### BWA
#########
RUN wget http://downloads.sourceforge.net/project/bio-bwa/bwa-0.7.15.tar.bz2 && \
	tar -xf bwa-0.7.15.tar.bz2
WORKDIR bwa-0.7.15
RUN make && \
	cp bwa /usr/local/bin/

#########
### Pip installs: MACS2
#########
# PySam
RUN pip install MACS2
WORKDIR /build

#########
### USeq and Sissrs
#########
RUN mkdir /data && \
    wget http://downloads.sourceforge.net/project/useq/USeq_8.9.6.zip && \
    unzip USeq_8.9.6.zip && mv USeq_8.9.6 /usr/share/USeq && \
    wget http://dir.nhlbi.nih.gov/papers/lmi/epigenomes/sissrs/sissrs_v1.4.tar.gz && \
    tar -xf sissrs_v1.4.tar.gz && \
    cp sissrs.pl /usr/local/bin

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
