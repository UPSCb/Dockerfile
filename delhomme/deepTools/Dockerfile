FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y build-essential locales python3-dev wget

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
### conda
#########
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN echo "1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f Miniconda3-py39_4.10.3-Linux-x86_64.sh" > SHA256.txt
RUN sha256sum -c SHA256.txt
RUN bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p /conda

#########
### deepTools
#########
RUN /conda/bin/conda install -c conda-forge -c bioconda deepTools

#########
### Cleanup
#########
WORKDIR /
RUN rm -rf build

#########
### deepTools
#########
ENTRYPOINT ["/conda/bin/deeptools"]
CMD ["-h"]

