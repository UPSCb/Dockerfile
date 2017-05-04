FROM delhomme/upscb-jbrowse
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt update && apt install -y bzip2 wget git python2.7-dev python-pip \
    openssl openssh-server libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    ea-utils

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
### Clean up
#########
WORKDIR /
RUN rm -rf /build /home/training/build

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
