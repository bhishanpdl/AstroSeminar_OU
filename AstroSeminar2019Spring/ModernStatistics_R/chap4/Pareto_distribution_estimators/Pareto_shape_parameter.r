#!/usr/bin/Rscript 

# Author : Bhishan Poudel 
# Date   : Feb 9, 2016
# Program: 

# Set the working directory  
this.dir <- dirname(parent.frame(2)$ofile) 
setwd(this.dir)  

# Start device driver to save output
#postscript(file= "Pareto_shape_parameter.eps", height = 14, width = 8)

# Estimates of Pareto shape parameter (power-law slope)
npts <- 100
alpha.MLE <- function(x) 1 / mean(log(x))
alpha.MVUE <- function(x) (1-(2/npts)) / mean(log(x))
alpha.mom <- function(x) 1 / (1-1/mean(x))
alpha.LS.EDF <- function(x) {
    lsd <- log(sort(x)) ; lseq <- log((npts:1) / npts)
    tmp <- lm(lseq~lsd)
    # plot(lsd, lseq) ; abline(tmp,col=2)
    -tmp$coef[2] }
alpha.LS.hist <- function(x) {
    hx <- hist(x, nclass=20, plot=F)
    counts <- hx$counts
    ldens <- log(hx$density[counts>0])
    lmidpts <- log(hx$mids[counts>0])
    tmp1 <- lm(ldens~lmidpts)
    # plot(lmidpts, ldens) ; abline(tmp1, col=2)
    alpha.LS.hist <- -1 - as.numeric(tmp1$coef[2])
    return(alpha.LS.hist) }
alpha.LS.hist.wt <- function(x) {
    hx <- hist(x, nclass=20, plot=F)
    counts <- hx$counts
    ldens <- log(hx$density[counts>0])
    lmidpts <- log(hx$mids[counts>0])
    tmp2 <- lm(ldens~lmidpts, weights=counts[counts>0])
    alpha.LS.hist.wt <- -1 - as.numeric(tmp2$coef[2])
    return(alpha.LS.hist.wt) }
six_alphas <- function(x) {
    out <- c(alpha.MLE(x), alpha.MVUE(x), alpha.mom(x),
             alpha.LS.EDF(x), alpha.LS.hist(x), alpha.LS.hist.wt(x))
    return(out) }

# Turn off device driver 
#dev.off() 


