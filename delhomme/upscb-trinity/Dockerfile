FROM delhomme/upscb-ngs-preprocessing
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN	apt-get install -y apt-utils sqlite3 libghc-zlib-dev libcurl4-openssl-dev

#########
### Working dir
#########
RUN	mkdir build
WORKDIR /build

#########
### Perl package
#########
RUN cpan App::cpanminus && cpanm URI::Escape

#########
### HTSlib tools: Samtools, Bcftools
#########
WORKDIR /build
RUN wget https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2 \
	https://github.com/samtools/htslib/releases/download/1.9/htslib-1.9.tar.bz2 && \
	tar -xf htslib-1.9.tar.bz2
WORKDIR htslib-1.9
RUN ./configure && make && make install
WORKDIR /build
RUN tar -xf samtools-1.9.tar.bz2
WORKDIR samtools-1.9
RUN make && make install
WORKDIR /build

#########
### bowtie 2
#########
RUN wget https://sourceforge.net/projects/bowtie-bio/files/bowtie2/2.3.5.1/bowtie2-2.3.5.1-linux-x86_64.zip && \
    unzip bowtie2-2.3.5.1-linux-x86_64.zip && find bowtie2-2.3.5.1-linux-x86_64 -executable -type f -exec cp "{}" /usr/local/bin \;

#########
### Jellyfish
#########
RUN wget https://github.com/gmarcais/Jellyfish/releases/download/v2.3.0/jellyfish-2.3.0.tar.gz && \
    tar -zxf jellyfish-2.3.0.tar.gz && cd jellyfish-2.3.0 && ./configure && make -j 4 && make install && \
    ldconfig

RUN apt install -y cmake automake libboost-all-dev

#########
### Salmon
#########
WORKDIR /build
RUN wget https://github.com/COMBINE-lab/salmon/archive/v0.14.2.tar.gz && \
    tar -xf v0.14.2.tar.gz && \
    cd /build/salmon-0.14.2 && mkdir build && cd build && \
    cmake .. && make -j 4 && make install && cp ../lib/lib* /usr/local/lib/. && \
    cp ../bin/salmon /usr/local/bin/.

#########
### trinity
#########
WORKDIR /opt
RUN wget https://github.com/trinityrnaseq/trinityrnaseq/releases/download/v2.8.6/trinityrnaseq-v2.8.6.FULL.tar.gz && \
  tar -zxf trinityrnaseq-v2.8.6.FULL.tar.gz && rm trinityrnaseq-v2.8.6.FULL.tar.gz && \
  cd trinityrnaseq-v2.8.6 && make && make plugins && \
  echo "export TRINITY_HOME=/opt/trinityrnaseq-v2.8.6" >> /home/training/.bashrc && \
    echo 'export PATH=/opt/trinityrnaseq-v2.8.6:$PATH' >> /home/training/.bashrc
WORKDIR /build

#########
### kallisto
#########
RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.46.0/kallisto_linux-v0.46.0.tar.gz && \
	tar -xf kallisto_linux-v0.46.0.tar.gz && rm kallisto_linux-v0.46.0.tar.gz
WORKDIR kallisto
RUN cp kallisto /usr/local/bin
WORKDIR /build

#########
### BLAST+
#########
RUN curl -O ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.9.0+-x64-linux.tar.gz && \
	tar xvf ncbi-blast-2.9.0+-x64-linux.tar.gz && \
	cp ncbi-blast-2.9.0+/bin/* /usr/local/bin/

#########
### Transdecoder
#########
WORKDIR /opt
RUN wget https://github.com/TransDecoder/TransDecoder/archive/TransDecoder-v5.5.0.tar.gz && \
  tar -xvzf TransDecoder-v5.5.0.tar.gz && rm TransDecoder-v5.5.0.tar.gz && \
  cd TransDecoder-TransDecoder-v5.5.0 && make && \
  echo "export TRANSDECODER_HOME=/opt/TransDecoder-TransDecoder-v5.5.0" >> /home/training/.bashrc && \
  echo 'export PATH=/opt/TransDecoder-TransDecoder-v5.5.0:$PATH' >> /home/training/.bashrc

#########
### SignalP
#########
#WORKDIR /opt
#ADD signalp-4.1e.Linux.tar.gz /opt/
#RUN sed -i "s:/usr/cbs/bio/src/signalp-4.1:/opt/signalp-4.1:" /opt/signalp-4.1/signalp && \
#  echo 'export PATH=/opt/signalp-4.1:$PATH' >> /home/training/.bashrc

#########
### tmHMM
#########
#ADD tmhmm-2.0c.Linux.tar.gz /opt/
#RUN echo 'export PATH=/opt/tmhmm-2.0c/bin:$PATH' >> /home/training/.bashrc && \
#  cd /usr/local/bin && ln -s $(which perl) .

#########
### hmmer3
#########
WORKDIR /build
RUN wget http://eddylab.org/software/hmmer3/3.1b2/hmmer-3.1b2-linux-intel-x86_64.tar.gz && \
  tar -zxf hmmer-3.1b2-linux-intel-x86_64.tar.gz && cp hmmer-3.1b2-linux-intel-x86_64/binaries/* /usr/local/bin

#########
### trinnotate
#########
WORKDIR /opt
RUN wget https://github.com/Trinotate/Trinotate/archive/Trinotate-v3.2.0.tar.gz && \
  tar -zxf Trinotate-v3.2.0.tar.gz && rm Trinotate-v3.2.0.tar.gz && \
  cd Trinotate-Trinotate-v3.2.0 && \
  echo "export TRINOTATE_HOME=/opt/Trinotate-Trinotate-v3.2.0" >> /home/training/.bashrc && \
  echo 'export PATH=/opt/Trinotate-Trinotate-v3.2.0:$PATH' >> /home/training/.bashrc

#########
### TrinnotateWeb
#########
WORKDIR /var/www/html
ADD cgi-enabled.conf /etc/apache2/conf-available/
RUN a2enmod cgi && cpanm DBI CGI HTML::Template && \
  ln -s /opt/Trinotate-3.2.0/TrinotateWeb . && a2enconf cgi-enabled

#########
### python dependencies
#########
RUN pip install numpy

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
