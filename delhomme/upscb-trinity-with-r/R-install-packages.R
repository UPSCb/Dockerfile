source("http://bioconductor.org/biocLite.R")
biocLite(scan(what="character","R-package-list.txt"),ask = FALSE)
biocValid(fix=TRUE,ask=FALSE)
