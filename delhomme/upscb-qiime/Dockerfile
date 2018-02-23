FROM delhomme/upscb-jbrowse
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### tmux med group
#########
#RUN groupadd -g 110 med && usermod -a -G med training

#########
### Aptitude packages
#########
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y bzip2 wget git python2.7-dev python-pip \
    openssl openssh-server libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    med-config ea-utils nano screen tmux

#########
### Anaconda
#########
WORKDIR /home/training/build
RUN wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh && \
    su -l training -c 'bash ~/build/Anaconda2-4.3.1-Linux-x86_64.sh -b' && \
    echo "export PATH=/home/training/anaconda2/bin:$PATH" >> /home/training/.bashrc

RUN su -l training -c '/home/training/anaconda2/bin/conda create -y -n lint python=2.7 && \
        /home/training/anaconda2/bin/conda install -y -n lint numpy scipy pylint matplotlib'

#########
### QIIME
#########
RUN su -l training -c '/home/training/anaconda2/bin/conda create -n qiime1 python=2.7 qiime \
matplotlib=1.4.3 mock nose -c bioconda'

#########
### GateOne SSH interface
#########
RUN pip install tornado==4.4.3
RUN git clone https://github.com/liftoff/GateOne/ && \
    cd GateOne && python setup.py install && \
    python run_gateone.py --configure && cd ..

#########
### Screed
#########
RUN pip install screed

#########
### Blast
#########
RUN curl -O ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.7.1+-x64-linux.tar.gz && \
	tar xvf ncbi-blast-2.7.1+-x64-linux.tar.gz && \
	cp ncbi-blast-2.7.1+/bin/* /usr/local/bin/

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build /home/training/build

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
