FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN apt install -y build-essential git nodejs

#########
### clone the repo
#########
RUN git clone https://autodocker:autodocker@microasp.upsc.se/bastian/course-web.git

#########
### PORT and CMD
#########
EXPOSE 3000

# For the course to be able to update the nodejs webpage daily
RUN apt install -y supervisor
#ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
#CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
CMD ["nodejs","/course-web/bin/www"]


