FROM delhomme/upscb-r-base
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
# RStudio relies ona libssl version 1.0.0
RUN 	echo "deb http://security.ubuntu.com/ubuntu xenial-security main" >> /etc/apt/sources.list
RUN     apt update
RUN	apt install -y gdebi-core sudo libssl1.0.0 && \
	mkdir build
WORKDIR /build

#########
### Rstudio
#########
RUN wget https://download2.rstudio.org/server/xenial/amd64/rstudio-server-1.3.1093-amd64.deb && \
    gdebi -n rstudio-server-1.3.1093-amd64.deb

#########
### Cleanup
#########
WORKDIR /
RUN rm -rf build

#########
### supervisord
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
EXPOSE 8787
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
