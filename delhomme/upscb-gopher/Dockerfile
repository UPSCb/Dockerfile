FROM ubuntu
MAINTAINER Nicolas Delhomme (nicolas.delhomme@umu.se)

#########
### Aptitude packages
#########
RUN apt-get update
RUN apt install -y --fix-missing build-essential g++ git make zlib1g-dev libboost-all-dev libssl-dev cmake

#########
### clone the cpprest repos
#########
WORKDIR /build
RUN git clone https://github.com/Microsoft/cpprestsdk.git casablanca
RUN cd casablanca/Release && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && make install

#########
### update cmake
#########
WORKDIR /build
RUN apt install wget && wget https://cmake.org/files/v3.11/cmake-3.11.1.tar.gz && tar -zxf cmake-3.11.1.tar.gz && \
cd cmake-3.11.1 && cmake . && make && make install

#########
### clone the repo
#########
WORKDIR /build
RUN git clone https://autodocker:autodocker@microasp.upsc.se/bastian/gopher2.git
RUN cd gopher2/build && rm -rf * && cmake -DBOOST_ROOT=/usr/lib/x86_64-linux-gnu .. && make && cp gopher2 /usr/local/bin

#########
### cleanup
#########
WORKDIR /
RUN rm -rf /build

#########
### PORT and CMD
#########
EXPOSE 5432

# For the course to be able to update the nodejs webpage daily
# RUN apt install -y supervisor
# ADD supervisord.conf /etc/supervisor/conf.d/supervisord.conf
# CMD ["/usr/bin/supervisord","-c","/etc/supervisor/conf.d/supervisord.conf"]
CMD ["gopher2","/usr/local/bin"]


