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
RUN apt update && apt install -y liblzma-dev libbz2-dev cmake automake curl libboost-all-dev

#########
### HTSlib tools: Samtools, Bcftools
#########
RUN wget https://github.com/samtools/samtools/releases/download/1.4/samtools-1.4.tar.bz2 \
	https://github.com/samtools/bcftools/releases/download/1.4/bcftools-1.4.tar.bz2 \
	https://github.com/samtools/htslib/releases/download/1.4/htslib-1.4.tar.bz2 && \
	tar -xf htslib-1.4.tar.bz2
WORKDIR htslib-1.4
RUN ./configure && make && make install
WORKDIR /build
RUN tar -xf bcftools-1.4.tar.bz2
WORKDIR bcftools-1.4
RUN make && make install
WORKDIR /build
RUN tar -xf samtools-1.4.tar.bz2
WORKDIR samtools-1.4
RUN make && make install
WORKDIR /build

#########
### STAR
#########
RUN wget https://github.com/alexdobin/STAR/archive/2.5.3a.tar.gz && \
	tar -xf 2.5.3a.tar.gz
WORKDIR STAR-2.5.3a/bin/Linux_x86_64_static
RUN cp STAR /usr/local/bin
WORKDIR /build

#########
### kallisto
#########
RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.43.1/kallisto_linux-v0.43.1.tar.gz && \
	tar -xf kallisto_linux-v0.43.1.tar.gz
WORKDIR kallisto_linux-v0.43.1
RUN cp kallisto /usr/local/bin
WORKDIR /build

#########
### Salmon
#########
WORKDIR /build
RUN wget https://github.com/COMBINE-lab/salmon/archive/v0.8.2.tar.gz && \
    tar -xf v0.8.2.tar.gz && \
    cd /build/salmon-0.8.2 && mkdir build && cd build && \
    cmake .. && make -j 4 && make install && cp ../lib/lib* /usr/local/lib/. && \
    cp ../bin/salmon /usr/local/bin/.

#########
### Pip installs: HTSeq
#########
# PySam
RUN pip install htseq

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]

