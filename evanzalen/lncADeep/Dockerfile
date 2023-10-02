FROM python:2.7-slim
#MAINTAINER Elena van Zalen (elena.vanzalen@umu.se)

# Aptitude packages
RUN apt update && apt install --fix-missing -y build-essential python-pip git

# Python dependicies
RUN pip install numpy==1.13.1 Theano==0.9.0 pandas==0.20.3 h5py==2.7.0

# lncMachine
WORKDIR /usr/local
RUN git clone https://github.com/cyang235/LncADeep.git

# R

# Entry point
CMD [ "python", "/usr/local/lncADeep/src/lncADeep.py" ]
#CMD [ "python", "hello.py" ]

