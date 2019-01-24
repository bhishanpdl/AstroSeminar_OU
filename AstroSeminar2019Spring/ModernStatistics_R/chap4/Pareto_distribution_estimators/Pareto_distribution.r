#!/usr/bin/env Rscript

# Author  : Bhishan Poudel
# Date    : Feb 9, 2016
# Program : 

# setting working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Start device driver to save output
#postscript(file="Pareto_distribution.eps", height=14, width=8)

# Define Pareto distributions for Salpeter IMF
dpareto <- function(x, alpha=1.35, b=1) (alpha>0)*(b>0)^alpha / x^(alpha+1)
ppareto <- function(x, alpha=1.35, b=1) (x>b)*(1-((b>0)/x)^(alpha>0))
qpareto <- function(u, alpha=1.35, b=1) (b>0)/(1-u)^(1/(alpha>0)) # 0<u<1
rpareto <- function(n, alpha=1.35, b=1) qpareto(runif(n),alpha,b)

# Produce Pareto-distributed simulated dataset with n=500; plot
par(mfrow=c(1,3))
z <- rpareto(500)

dotchart ( log10(z), main='', xlab='log stellar mass', pch=20, cex.lab=1.5 )

x <- seq(1, 50, len=1000)
plot (log10(x), log10(dpareto(x)), type="l", ylab='', xlab='log stellar mass', 
     cex.lab=1.5, cex.axis=1.5)
lines(log10(x), log10(ppareto(x)), type="l", lty=2)
legend (0.4, -0.3, legend=c('log(p.d.f)', 'log(c.d.f)'), lty=c(1,2), cex=1.5)

u=seq(.0005, .9095, len=1000)
plot (log10(qpareto(u)), u, type="l", xlab='log stellar mass', ylab='Quantile',
     cex.lab=1.5, cex.axis=1.5)
par(mfrow=c(1,1))

# Turn off device driver
#dev.off()


