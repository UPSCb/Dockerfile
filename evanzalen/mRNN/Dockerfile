FROM ubuntu:20.04
#MAINTAINER Elena van Zalen (elena.vanzalen@umu.se)

# Aptitude packages
RUN apt update && apt install --fix-missing -y build-essential python3-pip git

# Python dependicies
RUN pip3 install biopython optparse-pretty numpy pandas scikit-learn==0.22 pickle5

# lncMachine
WORKDIR /usr/local
RUN git clone https://github.com/hbusra/lncMachine.git

# R

# Entry point
CMD [ "python3", "/usr/local/lncMachine/src/lncMachine.py" ]
#CMD [ "python3", "hello.py" ]
