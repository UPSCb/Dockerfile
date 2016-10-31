FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)
RUN groupadd -g 2000 training && useradd -m -u 2000 -g 2000 training
RUN echo 'training:training' | chpasswd
RUN chsh -s /bin/bash training
RUN echo "alias ll='ls -la -G'" >> /home/training/.profile

#########
### Aptitude packages
#########
RUN     apt-get update
RUN	apt-get install -y \
	build-essential wget libghc-zlib-dev libncurses-dev libbz2-dev liblzma-dev libpcre3-dev libxml2-dev \
	libblas-dev gfortran git unzip ftp libzmq3-dev nano ftp fort77 libreadline-dev libcurl4-openssl-dev \
	libx11-dev libxt-dev x11-common libcairo2-dev libpng12-dev libreadline6-dev libjpeg8-dev pkg-config && \
	mkdir build
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
### Cleanup
#########
WORKDIR /
RUN rm -rf build

#########
### R installs
#########
ENV R_LIBS="/home/training/.r-library"
ADD .Renviron /home/training/.Renviron
RUN mkdir /home/training/.r-library && \
    chown -R training:training /home/training/.Renviron && \
    chown -R training:training /home/training/.r-library && \
    chmod 755 /home/training/.r-library

#########
### supervisord
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir /home/training/share
RUN apt-get install -y supervisor
VOLUME /home/training/share
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
