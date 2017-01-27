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
CMD ["nodejs","/course-web/bin/www"]


