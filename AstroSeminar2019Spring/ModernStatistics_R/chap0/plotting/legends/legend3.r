#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016

# set up a device driver to plot
#postscript(file='~/Copy/Programming/R/rprograms/plotting/legends/legend3.eps')

par(mar = c(4, 4, 2, 0.1))
plot(rnorm(100), rnorm(100),
     xlab = expression(hat(mu)[0]), ylab = expression(alpha^beta),
     main = expression(paste("Plot of ", alpha^beta, " versus ", hat(mu)[0])))

# turn off device driver
#dev.off()