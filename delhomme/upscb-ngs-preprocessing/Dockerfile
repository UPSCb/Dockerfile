FROM delhomme/upscb-jbrowse
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update && apt-get install -y default-jre wget git ftp nano python2.7-dev python-pip openssl openssh-server \
	libcairo2-dev libpango1.0-dev

#########
### Working dir
#########
RUN	mkdir build 
WORKDIR /build

#########
### Java tools: Trimmomatic, FastQC
#########
RUN wget http://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.11.5.zip && \
	unzip fastqc_v0.11.5.zip && \
	chmod +x FastQC/fastqc && \
	cp -r FastQC /usr/share/ && \
	ln -s /usr/share/FastQC/fastqc /usr/bin/ && \
	wget http://www.usadellab.org/cms/uploads/supplementary/Trimmomatic/Trimmomatic-0.36.zip && \
	unzip Trimmomatic-0.36.zip && \
	cp -r Trimmomatic-0.36 /usr/share/ && \
	echo '#!/bin/bash\njava -jar /usr/share/Trimmomatic-0.36/trimmomatic-0.36.jar $@' > /usr/bin/trimmomatic && \
	chmod +x /usr/bin/trimmomatic

#########
### SortmeRNA
#########
RUN wget https://github.com/biocore/sortmerna/archive/2.1.tar.gz && \
    tar -xf 2.1.tar.gz
WORKDIR sortmerna-2.1
RUN ./configure && make && make install || true
RUN cp indexdb_rna /usr/local/bin && \
  cp scripts/merge-paired-reads.sh /usr/local/bin && cp scripts/unmerge-paired-reads.sh /usr/local/bin && \
	cp -r rRNA_databases/ /usr/share/
WORKDIR /usr/share/rRNA_databases
ADD scripts/phiX.fasta /usr/share/rRNA_databases/phiX.fasta
COPY scripts/indexdb.sh /usr/share/rRNA_databases/
COPY scripts/makeDbList.sh /usr/share/rRNA_databases/
RUN ./indexdb.sh
ENV SORTMERNA_DB="/usr/share/rRNA_databases/rfam-5s-database-id98.fasta,/usr/share/rRNA_databases/index/rfam-5s-database-id98:/usr/share/rRNA_databases/silva-euk-28s-id98.fasta,/usr/share/rRNA_databases/index/silva-euk-28s-id98:/usr/share/rRNA_databases/silva-euk-18s-id95.fasta,/usr/share/rRNA_databases/index/silva-euk-18s-id95:/usr/share/rRNA_databases/silva-arc-16s-id95.fasta,/usr/share/rRNA_databases/index/silva-arc-16s-id95:/usr/share/rRNA_databases/silva-bac-16s-id90.fasta,/usr/share/rRNA_databases/index/silva-bac-16s-id90:/usr/share/rRNA_databases/silva-arc-23s-id98.fasta,/usr/share/rRNA_databases/index/silva-arc-23s-id98:/usr/share/rRNA_databases/silva-bac-23s-id98.fasta,/usr/share/rRNA_databases/index/silva-bac-23s-id98:/usr/share/rRNA_databases/rfam-5.8s-database-id98.fasta,/usr/share/rRNA_databases/index/rfam-5.8s-database-id98:/usr/share/rRNA_databases/phiX.fasta,/usr/share/rRNA_databases/index/phiX"
RUN echo "export SORTMERNA_DB=$SORTMERNA_DB" >> /home/training/.profile
WORKDIR /build

#########
### Pip installs: multiQC
#########
WORKDIR /build
RUN git clone https://github.com/ewels/MultiQC && cd MultiQC && \
    pip install numpy matplotlib && \
    python setup.py install

WORKDIR /build
RUN wget http://genometools.org/pub/genometools-1.5.9.tar.gz && \
    tar -xf genometools-1.5.9.tar.gz && cd genometools-1.5.9 && \
    make && make install

#########
### GateOne SSH interface
#########
RUN pip install tornado==4.4.3
RUN git clone https://github.com/liftoff/GateOne/ && \
    cd GateOne && python setup.py install && \
    python run_gateone.py --configure && cd ..

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
# Apache access to the home dir
#########
RUN ln -sf /home/training /var/www/html/home

#########
### Supervisor
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf

#########
### Volumes
#########
RUN mkdir /home/training/share
VOLUME /home/training/share

#########
### Ports and CMD
#########
EXPOSE 22 80 443
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
