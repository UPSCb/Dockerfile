rlib='/home/training/.r-library'

install.packages("BiocManager",repos="https://ftp.acc.umu.se/mirror/CRAN/")
BiocManager::install(
	pkgs=scan(what="character","/home/training/R-package-list.txt"),
	ask = FALSE,
	update=TRUE,
	Ncpus=8)
