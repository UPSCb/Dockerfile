FROM delhomme/upscb-r-base
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

# Version
ARG RST_VERSION=2023.03.0-386

# Aptitude packages
# RUN     echo "deb http://security.ubuntu.com/ubuntu xenial-security main" >> /etc/apt/sources.list
RUN     apt update
#RUN	apt install -y gdebi-core && \
RUN	apt install -y gdebi-core sudo libssl-dev && \
	mkdir build
WORKDIR /build

#########
### Rstudio
#########
RUN curl -L -O https://download2.rstudio.org/server/jammy/amd64/rstudio-server-${RST_VERSION}-amd64.deb && \
	gdebi -n rstudio-server-${RST_VERSION}-amd64.deb

# Cleanup
WORKDIR /
RUN rm -rf build

# Config
ADD rserver.conf /etc/rstudio/rserver.conf
ADD rsession.conf /etc/rstudio/rsession.conf

# supervisord
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# port
EXPOSE 8787

# cmd
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
