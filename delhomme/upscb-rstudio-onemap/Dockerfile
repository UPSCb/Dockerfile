FROM delhomme/upscb-rstudio
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update && apt install -y libssh2-1-dev autoconf

#########
# R packages
#########
WORKDIR /home/training
ADD R-package-list.txt /home/training/R-package-list.txt
ADD R-install-packages.R /home/training/R-install-packages.R
ADD .Renviron /home/training/.Renviron
RUN Rscript R-install-packages.R

########
# GitHub package
########
RUN Rscript -e 'devtools::install_github("augusto-garcia/onemap")'

#########
### Cleanup
#########
WORKDIR /home/training
RUN rm -rf R-package-list.txt R-install-packages.R
WORKDIR /
RUN rm -rf build

#########
### supervisord
#########
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
