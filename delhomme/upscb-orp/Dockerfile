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
RUN apt update && apt install -y automake autotools-dev cmake emboss hmmer infernal libboost-all-dev \
libfontconfig1 libsm6 liburi-escape-xs-perl liburi-perl libxrender1 mcl ncbi-blast+ parallel python-numpy \
ruby unzip && gem install crb-blast

#########
### LinuxBrew
#########
RUN mkdir /home/linuxbrew && chown training:training /home/linuxbrew
USER training
RUN git clone https://github.com/Linuxbrew/brew.git /home/linuxbrew/.linuxbrew
ENV PATH=/home/linuxbrew/.linuxbrew/bin:${PATH} MANPATH=/home/linuxbrew/.linuxbrew/share/man:${MANPATH} INFOPATH=/home/linuxbrew/.linuxbrew/share/info:${INFOPATH}
RUN echo "PATH=/home/linuxbrew/.linuxbrew/bin:${PATH}\nMANPATH=/home/linuxbrew/.linuxbrew/share/man:${MANPATH}\nINFOPATH=/home/linuxbrew/.linuxbrew/share/info:${INFOPATH}" >> /home/training/.profile
RUN echo "PATH=/home/linuxbrew/.linuxbrew/bin:${PATH}\nMANPATH=/home/linuxbrew/.linuxbrew/share/man:${MANPATH}\nINFOPATH=/home/linuxbrew/.linuxbrew/share/info:${INFOPATH}" >> /home/training/.bash_profile
RUN brew tap brewsci/science && brew tap brewsci/bio && brew update && brew install gcc python metis parallel
USER root

#########
### Python modules
#########
RUN pip install cvxopt numpy biopython scipy

#########
### Oyster River Protocol (ORP)
#########
WORKDIR /home/training
USER training
RUN git clone https://github.com/macmanes-lab/Oyster_River_Protocol.git
RUN cd Oyster_River_Protocol && make
RUN mkdir Oyster_River_Protocol/busco_dbs && cd $HOME/Oyster_River_Protocol/busco_dbs && \
wget http://busco.ezlab.org/v2/datasets/eukaryota_odb9.tar.gz && tar -zxf eukaryota_odb9.tar.gz
ADD config.ini /home/training/Oyster_River_Protocol/software/busco/config/config.ini
USER root

#########
### Miniconda
#########
USER training
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
bash miniconda.sh -b -p /home/training/miniconda

#########
### Path
#########
RUN echo 'PATH=$PATH:/home/training/miniconda/bin' >> /home/training/.profile
RUN echo 'PATH=$PATH:/home/training/miniconda/bin' >> /home/training/.bash_profile
ENV PATH=/home/training/miniconda/bin:${PATH}

#########
### Dammit
#########
RUN conda update -n base conda && conda create -n dammit python=3
RUN /bin/bash -c "source activate dammit && \
	conda config --add channels conda-forge && \
	conda config --add channels bioconda"
RUN /bin/bash -c "conda install -y dammit"
# || conda install -y dammit"

#########
### Databases
#########
RUN /bin/bash -c "source activate dammit && \
	dammit databases --install && \
	dammit databases --install --busco-group fungi"
ADD config.ini /home/training/Oyster_River_Protocol/software/config.ini
USER root

#########
### Clean up
#########
WORKDIR /
RUN rm -rf /build

#########
### Ports and CMD
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
