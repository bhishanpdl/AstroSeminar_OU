#!/usr/bin/env Rscript

# Author  : Bhishan Poudel
# Date    : Feb 11, 2016
# Program :

# Setting working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Start device driver to save output
#postscript( file = "asteroid.eps", height = 12, width = 8)

# Data plots and summaries for asteroid densities
asteroids <- read.table("asteroid_dens.dat", header = T)
astnames <-
    asteroids[, 1]
dens <- asteroids[, 2]
err <- asteroids[, 3]
dotchart(dens,
         labels = astnames,
         cex = 0.9,
         xlab = expression(Density ~ (g / cm ^ 3)))
plot(
    dens,
    ylim = c(0, 8),
    xlab = "Asteroids",
    ylab = expression(Density ~ (g / cm ^ 3)),
    pch = 20
)
num <- seq(1, length(dens))
segments(num, dens + err, num, dens - err)
# Boxplot and summary statistics for asteroid data
boxplot(
    asteroids[, 2:3],
    varwidth = T,
    notch = T,
    xlab = 'Asteroids',
    ylab = 'Density',
    pars = list(
        boxwex = 0.3,
        boxlwd = 1.5,
        whisklwd = 1.5,
        staplelwd = 1.5,
        outlwd = 1.5,
        font = 2
    )
)
summary(dens)
mean(dens)
sd(dens)
median(dens)
mad(dens)
weighted.mean(dens, 1 / err ^ 2)

# Tests for normality
shapiro.test(dens)
#install.packages('nortest')
library('nortest')
ad.test(dens)
cvm.test(dens)
lillie.test(dens)
pearson.test(dens)

#install.packages('outliers')
library('outliers')
dixon.test(dens)
chisq.out.test(dens)
grubbs.test(dens)
grubbs.test(dens, type = 20)

# Turn off device driver
#dev.off()

