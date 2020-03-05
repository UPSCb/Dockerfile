FROM bschiffthaler/seidr:devel
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN dnf update -y && dnf install -y httpd shellinabox supervisor

#########
## create the User
#########
RUN groupadd -g 2000 training && useradd -m -u 2000 -g 2000 training
RUN echo "alias ll='ls -la -G'" >> /home/training/.profile
RUN	usermod -G training,apache training

#########
### accessibility
#########
RUN ln -sf /home/training /var/www/html/home
COPY httpd.conf /etc/httpd/conf/httpd.conf
RUN rm /etc/httpd/conf.d/welcome.conf
RUN mkdir /home/training/share
VOLUME /home/training/share
EXPOSE 22 80 4200

#########
### supervisord
#########
ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
ADD pwgen /bin/pwgen
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
