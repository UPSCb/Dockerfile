FROM delhomme/upscb-rstudio
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update && apt install -y curl libssh2-1-dev librsvg2-dev libssl-dev imagemagick

#########
# R packages
#########
WORKDIR /home/training
ADD R-package-list.txt /home/training/R-package-list.txt
ADD R-install-packages.R /home/training/R-install-packages.R
ADD .Renviron /home/training/.Renviron
RUN Rscript R-install-packages.R

########
# R local packages
#########
#RUN	mkdir /build
#WORKDIR /build
#RUN curl ftp://130.239.72.87/Facility/Workshop/RnaSeqTutorial_0.99.14.tar.gz > RnaSeqTutorial_0.99.14.tar.gz
#RUN R CMD INSTALL RnaSeqTutorial_0.99.14.tar.gz

########
# GitHub package - kept as a reminder.
########
#RUN Rscript -e 'devtools::install_github("catavallejos/BASiCS")'

#########
### Cleanup
#########
#WORKDIR /home/training
RUN rm -rf R-package-list.txt R-install-packages.R
WORKDIR /
RUN rm -rf build

#RUN echo "training:L9NqpDEm" | chpasswd

#########
### supervisord
#########
#ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
