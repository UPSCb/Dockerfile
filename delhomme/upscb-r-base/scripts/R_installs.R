rlib='/home/training/.r-library'

source("http://bioconductor.org/biocLite.R")
biocLite(scan(what="character","R_packages.txt"),ask = FALSE)
biocValid(fix=TRUE,ask=FALSE)
