FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)
RUN groupadd -g 2000 training && useradd -m -u 2000 -g 2000 training
RUN chsh -s /bin/bash training
RUN echo "alias ll='ls -la -G'" >> /home/training/.profile

#########
### Aptitude packages
#########
RUN     apt-get update
RUN	apt-get install -y --fix-missing build-essential wget
# && \
RUN	DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
#&& \
RUN	apt-get install -y --fix-missing default-jdk ftp fort77 gfortran git libblas-dev libbz2-dev libcairo2-dev \
	libcurl4-openssl-dev libghc-zlib-dev libgl1-mesa-dev libglu1-mesa-dev libjpeg-dev libncurses-dev libopenblas-dev \
	libpcre3-dev libreadline-dev libudunits2-dev libx11-dev libxml2-dev libxt-dev liblzma-dev libzmq3-dev locales \
	mesa-common-dev nano pandoc pandoc-citeproc pkg-config supervisor tcl-dev tk-dev texlive-latex-base texinfo \
	texlive-fonts-extra unzip x11-common zip && \
	mkdir build
WORKDIR /build

#########
### Squash locale warnings
#########
RUN rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

#########
### R version 4.0.3
#########
# https://stat.ethz.ch/R/daily/R-devel.tar.gz && \
RUN wget http://ftp.acc.umu.se/mirror/CRAN/src/base/R-4/R-4.0.3.tar.gz && \
    tar -zxf R-4.0.3.tar.gz && cd R-4.0.3 && \
## RUN wget https://stat.ethz.ch/R/daily/R-devel.tar.gz && \
#    tar -zxf R-devel.tar.gz && cd R-devel && \
    ./configure --enable-memory-profiling -enable-R-shlib --with-blas --with-lapack --with-tcltk \
	--with-valgrind-instrumentation=3 --with-readline --with-cairo --with-libpng --with-jpeglib \
	--with-x=no && \
##    make && make pdf && make info && make install && make install-info && make install-pdf
## no test 2020/10/07
##	make -j 8 && make check && make -j 8 pdf && make -j 8 info && make -j 8 install && \
	make -j 8 && make -j 8 pdf && make -j 8 info && make -j 8 install && \
	make -j 8 install-info && make -j 8 install-pdf

#########
### Cleanup
#########
WORKDIR /
RUN rm -rf build

#########
### supervisord
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ADD pwgen /bin/pwgen

#########
### Volumes
#########
RUN mkdir /home/training/share
VOLUME /home/training/share

#########
### CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
