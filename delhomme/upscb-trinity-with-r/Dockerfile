FROM delhomme/upscb-trinity
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN	apt-get install -y apt-utils libcurl4-openssl-dev

#########
### Working dir
#########
RUN	mkdir build
WORKDIR /build

#########
### R version 3.4.3
#########
RUN apt-get update && apt install --fix-missing -y libncurses-dev libpcre3-dev libxml2-dev libblas-dev gfortran \
    unzip libzmq3-dev fort77 libreadline-dev libcurl4-openssl-dev \
    libx11-dev libxt-dev x11-common libpng12-dev libreadline6-dev \
    libjpeg8-dev pkg-config texlive-latex-base texinfo texlive-fonts-extra default-jdk \
    libghc-zlib-dev libbz2-dev liblzma-dev zip supervisor tcl-dev tk-dev

WORKDIR /build
RUN wget http://ftp.acc.umu.se/mirror/CRAN/src/base/R-3/R-3.4.3.tar.gz && \
    tar -zxf R-3.4.3.tar.gz && cd R-3.4.3 && \
        ./configure --enable-memory-profiling -enable-R-shlib --with-blas --with-lapack \
	        --with-valgrind-instrumentation=3 --with-readline --with-cairo --with-libpng --with-jpeglib --with-x=no && \
		          make && make pdf && make info && make install && make install-info && make install-pdf
			      ## make && make check && make pdf && make info && make install && make install-info && make install-pdf

#########
# R packages
#########
WORKDIR /home/training
ADD R-package-list.txt /home/training/R-package-list.txt
ADD R-install-packages.R /home/training/R-install-packages.R
RUN Rscript R-install-packages.R

#########
### RSEM
#########
WORKDIR /build
RUN wget https://github.com/deweylab/RSEM/archive/v1.3.0.tar.gz && tar -zxf v1.3.0.tar.gz && \
cd RSEM-1.3.0 && make && make install

#########
### Clean up
#########
WORKDIR /home/training
RUN rm -rf R-package-list.txt R-install-packages.R
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
