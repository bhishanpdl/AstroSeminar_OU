#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016

# set up a device driver to plot
#postscript(file='~/Copy/Programming/R/rprograms/plotting/legends/legend2.eps')

par(mar = c(4, 4, 2, 0.1))
x_mean <- 1.5
x_sd <- 1.2
hist( rnorm(100, x_mean, x_sd),
      main = substitute( paste(X[i], " ~ N(", mu, "=", m, ", ", sigma^2, "=", s2, ")"),
                         list(m = x_mean, s2 = x_sd^2)))

# turn off device driver
#dev.off()