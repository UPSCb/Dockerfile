FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN	apt install -y build-essential curl unzip apache2 libghc-zlib-dev

#########
### Squash locale warnings
#########
RUN locale-gen en_US.UTF-8

#########
### Working dir
#########
RUN	mkdir build 
WORKDIR /build

#########
## create the User
#########
RUN groupadd -g 2000 training && useradd -m -u 2000 -g 2000 training
RUN echo 'training:training' | chpasswd
RUN chsh -s /bin/bash training
RUN echo "alias ll='ls -la -G'" >> /home/training/.profile
RUN	usermod -G training,www-data training

#########
### JBrowse dep
#########
RUN cpan App::cpanminus && cpanm PerlIO::gzip

#########
### JBrowse to easily view biological data formats
#########
RUN rm -f /var/www/html/index.html
WORKDIR /var/www/html
RUN curl -O http://jbrowse.org/releases/JBrowse-1.12.1.zip && \
    unzip JBrowse-1.12.1.zip
WORKDIR /var/www/html/JBrowse-1.12.1
RUN ./setup.sh && \
    chown -R www-data:www-data /var/www/html
WORKDIR /
RUN rm /var/www/html/JBrowse-1.12.1.zip

#########
### Cleanup
#########
WORKDIR /
RUN rm -rf build

#########
### supervisord
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN apt-get install -y supervisor
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
