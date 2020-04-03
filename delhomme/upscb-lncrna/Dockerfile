FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update && apt-get upgrade -y
RUN	apt install -y build-essential dos2unix git libgsl-dev locales m4 opam \
      pkg-config python-pip python3-dev python3-pip unzip wget zip
RUN DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install r-base-core

#########
### Squash locale warnings
#########
RUN rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

#########
### Working dir
#########
RUN	mkdir build
WORKDIR /build

#########
### PLEK
#########
RUN wget https://downloads.sourceforge.net/project/plek/PLEK.1.2.tar.gz && \
  tar -zxf PLEK.1.2.tar.gz && cd PLEK.1.2 && ( python PLEK_setup.py || echo TRUE ) && \
  mkdir -p /opt/PLEK/bin && cp -r PLEK_spsn PLEK PLEK.range PLEK.model *.py svm-predict \
  svm-scale svm-train /opt/PLEK/bin && chmod +x /opt/PLEK/bin/*.py && dos2unix /opt/PLEK/bin/*.py

#########
### CNCI
#########
RUN git clone https://github.com/www-bioinfo-org/CNCI.git && cd CNCI && \
  unzip libsvm-3.0.zip && cd libsvm-3.0 && make && cd .. && rm libsvm-3.0.zip && \
  mkdir -p /opt/CNCI/bin && cp -r . /opt/CNCI/bin && chmod +x /opt/CNCI/bin/*.py /opt/CNCI/bin/*.pl

#########
### CPAT
#########
RUN apt-get update && apt-get upgrade -y && apt install -y zlib1g-dev libbz2-dev liblzma-dev
RUN pip3 install Cython && pip3 install pysam && pip3 install numpy && pip3 install CPAT

#########
### CPC
#########
RUN pip install biopython

ADD CPC2-beta.tar.gz /build

#RUN wget http://cpc2.cbi.pku.edu.cn/data/CPC2-beta.tar.gz && tar -zxf CPC2-beta.tar.gz && \

RUN cd CPC2-beta && cd libs/libsvm && tar -zxf libsvm-3.18.tar.gz && \
  cd libsvm-3.18 && make clean && make && cd .. && rm libsvm-3.18.tar.gz \
  && cd ../.. && mkdir /opt/CPC2 && cp -r . /opt/CPC2

#########
### PhyloCSF
#########
RUN opam init -a && opam install -y batteries ocaml-twt gsl

ENV PATH=/root/.opam/system/bin:${PATH}

RUN  git clone https://github.com/mlin/PhyloCSF.git && \
  cd PhyloCSF && make && mkdir -p /opt/PhyloCSF/bin && \
  cp -r PhyloCSF PhyloCSF.Linux.x86_64 PhyloCSF_Parameters /opt/PhyloCSF/bin

#########
### BLAST
#########
RUN wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.10.0+-x64-linux.tar.gz && \
        tar xvf ncbi-blast-2.10.0+-x64-linux.tar.gz && \
        cp ncbi-blast-2.10.0+/bin/* /usr/local/bin/

#########
### PLNCPRO
#########
RUN pip install regex scipy scikit-learn==0.17.1
RUN wget http://ccbb.jnu.ac.in/plncpro/downloads/plncpro_1.1.tar.gz && \
	tar -zxf plncpro_1.1.tar.gz && cd plncpro_1.1/lib/estate && make && \
	cp bin/framefinder ../framefinder && mkdir -p ../blast/bin && cp /build/ncbi-blast-2.10.0+/bin/* ../blast/bin

WORKDIR /build

RUN mv plncpro_1.1 /opt/plncpro

ADD blastparse_mt3.py /opt/plncpro/bin

ADD prediction.py /opt/plncpro

#########
### Diamond
#########
RUN wget http://github.com/bbuchfink/diamond/releases/download/v0.9.30/diamond-linux64.tar.gz && \
	tar xzf diamond-linux64.tar.gz && mv diamond /usr/local/bin/

#########
### PATH variable
#########
ENV PATH=/opt/plncpro/bin:/opt/PhyloCSF/bin:/opt/CPC2/bin:/opt/CNCI/bin:/opt/PLEK/bin:${PATH}

#########
### Cleanup
#########
WORKDIR /
RUN rm -rf build
