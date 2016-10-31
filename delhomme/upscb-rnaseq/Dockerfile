FROM delhomme/upscb-ngs-preprocessing
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)
	
#########
### Working dir
#########
RUN	mkdir build 
WORKDIR /build

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
### STAR
#########
RUN wget https://github.com/alexdobin/STAR/archive/2.5.2b.tar.gz && \
	tar -xf 2.5.2b.tar.gz
WORKDIR STAR-2.5.2b/bin/Linux_x86_64_static
RUN cp STAR /usr/local/bin
WORKDIR /build

#########
### kallisto
#########
RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.43.0/kallisto_linux-v0.43.0.tar.gz && \
	tar -xf kallisto_linux-v0.43.0.tar.gz
WORKDIR kallisto_linux-v0.43.0
RUN cp kallisto /usr/local/bin
WORKDIR /build

#########
### Pip installs: HTSeq
#########
# PySam
RUN	pip install htseq

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]

