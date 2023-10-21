FROM ubuntu:22.04
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

ARG COURSE=BUWS2301
#########
### Aptitude packages
#########
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y --fix-missing build-essential git nodejs npm

#########
### clone the repo
#########
RUN git clone -b ${COURSE} https://github.com/UPSCb/course-web.git

#########
### install the packages
#########
WORKDIR /course-web
RUN npm install

#########
### PORT and CMD
#########
EXPOSE 3000

#########
### materials
#########
VOLUME /course-web/public/materials

# For the course to be able to update the nodejs webpage daily
#RUN apt install -y supervisor
#ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
#CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
WORKDIR /
CMD ["nodejs","/course-web/bin/www"]
