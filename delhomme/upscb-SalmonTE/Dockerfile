FROM delhomme/upscb-r-base
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN     apt-get update
RUN	apt-get install --fix-missing -y python3-pip git && mkdir build

#########
### Python dependicies
#########
RUN pip3 install snakemake docopt pandas

#########
### R packages
#########
RUN Rscript -e 'install.packages(c("tidyverse", "scales", "WriteXLS", "BiocManager"),repos="https://ftp.acc.umu.se/mirror/CRAN/")' && \
  Rscript -e 'BiocManager::install("DESeq2", version = "3.8")'

#########
### SalmonTE
#########
WORKDIR /build
RUN git clone https://github.com/hyunhwaj/SalmonTE

RUN echo 'PATH=$PATH:/build/SalmonTE' >> /home/training/.bashrc
RUN echo 'PATH=$PATH:/build/SalmonTE' >> /home/training/.profile
ENV PATH=${PATH}:/build/SalmonTE


