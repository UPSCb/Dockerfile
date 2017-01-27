# Dockerfile

Host the dockerfile from dockerhub/delhomme

## Content

Sorted in dependencies order:

* Command line based:

1. upscb-jbrowse

2. upscb-ngs-preprocessing

3. upscb-chipseq and upscb-rnaseq

* R and R-studio based

1. upscb-r-base
2. upscb-rstudio
3. upscb-rstudio-sequencing

* Others:

1. upscb-nodejs

## Installation

If you do not have docker installed; do ( on a linux distro)
```
curl https://get.docker.com | sh
sudo usermod -a -G docker ubuntu
sudo service docker start
```

Then simply clone the present repository, cd into the docker you want to build
and do _e.g._:

```{bash}
git clone https://github.com/UPSCb/Dockerfile.git
cd delhomme/upscb-jbrowse
docker build -t delhomme/upscb-jbrowse
```

Or if you do not want to build it on your system, but just pull it

```{bash}
docker pull  delhomme/upscb-jbrowse
```

## Details

### Linux based containers

#### upscb-jbrowse
Create the user training from the group training with password
training and install JBrowse in /var/www/html. JBrowse is not
configured. To provide access to JBrowse you need to map the container
port 80; e.g. here is the run command that will start the docker and
map port 80 on port 9001

```{bash}
docker run -d -p 9001:80 delhomme/upscb-jbrowse
```

#### upscb-ngs-preprocessing
Build on the previous, contains all the tools necessary to run the
pre-processing of RNA-Seq data as described in our guidelines:
http://www.epigenesys.eu/en/protocols/bio-informatics/1283-guidelines-for-rna-seq-data-analysis.
The tools are:

* FastQC
* SortMeRNA
* Trimmomatic
* MultiQC

This container also installs GateOne, a terminal emulator for the web
browser, and expose the corresponding necessary port 443. It also
expose the ssh port 22.

The following command is an example to run it that maps 443 to 10001
and 22 to 11001.

```{bash}
docker run -d -p 9001:80 -p 10001:443 -p 11001:22 delhomme/upscb-jbrowse
```

#### upscb-chipseq and upscb-rnaseq

These two depends on the previous one and expose the same ports. They
contain tools necessary to do ChIP-Seq and RNA-Seq analysis
respectively, _i.e._

* R
* HTSlib
* BEDTools
* BWA
* MACS2
* USeq
* Sissrs

and

* HTSlib
* STAR
* HTSeq
* kallisto

respectively.

### R and R-studio based

1. upscb-r-base
   Simply install base R and create the user training from the group training with password
   training. The default R installation directory is configured to be /home/training/.r-library.

2. upscb-rstudio
   Install RStudio server (the non-commercial edition). It exposes port
   8787 (web browser access to RStudio). Here is the run command that
   will start the docker and map port 8787 on port 12001

```{bash}
docker run -d -p 12001:8787 delhomme/upscb-r-studio
```

3. upscb-rstudio-sequencing
   Build on the above, installing all the necessary R/Bioconductor
   packages for the analysis of RNA-Seq data.

## Others

1. upscb-nodejs

   A simple container containing a git checkout of our nodejs website used for courses and workshops.

## Some more docker commands

* to list the images
  `docker images`

* to remove images
  `docker rmi [image ID]`

* to list containers
  `docker ps -a`

* to stop a container
  `docker stop [container ID]`

* to remove a container
  `docker rm [container ID]`

* both in oneline
  `docker rm $(docker stop [container ID])`

* to connect to a container as root, activating some terminal support
  `docker exec -ti [container ID] env TERM=xterm bash`

