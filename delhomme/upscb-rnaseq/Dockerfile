FROM delhomme/upscb-ngs-preprocessing
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Working dir
#########
RUN	mkdir build
WORKDIR /build

#########
### Aptitude packages
#########
RUN apt update && apt install -y cmake automake curl libboost-all-dev

#########
### STAR
#########
RUN wget https://github.com/alexdobin/STAR/archive/2.6.1c.tar.gz && \
	tar -xf 2.6.1c.tar.gz
WORKDIR STAR-2.6.1c/bin/Linux_x86_64_static
RUN cp STAR /usr/local/bin
WORKDIR /build

#########
### kallisto
#########
RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.44.0/kallisto_linux-v0.44.0.tar.gz && \
	tar -xf kallisto_linux-v0.44.0.tar.gz
WORKDIR kallisto_linux-v0.44.0
RUN cp kallisto /usr/local/bin
WORKDIR /build

#########
### Salmon
#########
WORKDIR /build
RUN wget https://github.com/COMBINE-lab/salmon/archive/v0.11.3.tar.gz && \
    tar -xf v0.11.3.tar.gz && \
    cd /build/salmon-0.11.3 && mkdir build && cd build && \
    cmake .. && make -j 4 && make install && cp ../lib/lib* /usr/local/lib/. && \
    cp ../bin/salmon /usr/local/bin/.

#########
### Pip installs: HTSeq
#########
# PySam
RUN pip install htseq

#########
### gffread
#########
WORKDIR /build
RUN git clone https://github.com/gpertea/gclib && \
    git clone https://github.com/gpertea/gffread && \
    cd gffread && make && cp gffread /usr/local/bin

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
