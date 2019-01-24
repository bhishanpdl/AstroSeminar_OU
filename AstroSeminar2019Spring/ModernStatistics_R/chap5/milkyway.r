#!/usr/bin/env Rscript

# Author  : Bhishan Poudel
# Date    : Feb 11, 2016
# Program :

# Setting working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Start device driver to save output
#postscript( file = "milkyway.eps", height = 12, width = 8)

# Read magnitudes for Milky Way and M 31 globular clusters
GC1 <- read.table("GlobClus_MWG.dat",
                  header = T)
GC2 <- read.table("GlobClus_M31.dat",
                  header = T)
K1 <- GC1[, 2]
K2 <- GC2[, 2]
cat("summary of K1 \n")
print(summary(K1))
cat("\n summary of K1 \n")
print(summary(K2))

# Three estimates of the distance modulus to M 31
DMmn <- mean(K2) - mean(K1)
print(DMmn)
sigDMmn <- sqrt(var(K1) / length(K1) + var(K2) / length(K2))
print(sigDMmn)
DMmed <- median(K2) - median(K1)
print(DMmed)
sigDMmed <-
    sqrt(mad(K1) ^ 2 / length(K1) + mad(K2) ^ 2 / length(K2))
print(sigDMmed)
wilcox.test(K2, K1, conf.int = T)

# e.d.f., quantile and Q-Q plots for globular cluster magnitudes
plot(
    ecdf(K1),
    cex.points = 0,
    verticals = T,
    xlab = "K (mag)",
    ylab = "e.d.f.",
    main = ""
)
plot(
    ecdf(K2 - 24.90),
    cex.points = 0,
    verticals = T,
    add = T
)
text(-7.5, 0.8, lab = "MWG")
text(-10.5, 0.9, lab = "M 31")
par(mfrow = c(1, 3))
qqplot(
    K1,
    K2 - 24.90,
    pch = 20,
    cex.axis = 1.3,
    cex.lab = 1.3,
    xlab = "MWG",
    ylab = "M31 - 24.90",
    main = ""
)
qqnorm(
    K1,
    pch = 20,
    cex.axis = 1.3,
    cex.lab = 1.3,
    main = ""
)
qqline(K1, lty = 2, lwd = 1.5)
text(-2.5, -6, pos = 4, cex = 1.3,  'MWG normal QQ plot')
qqnorm(
    K2 - 24.90,
    pch = 20,
    cex.axis = 1.3,
    cex.lab = 1.3,
    main = ""
)
qqline(K2 - 24.90, lty = 2, lwd = 1.5)
text(-3, -7.5, pos = 4, cex = 1.3,  'M31 normal QQ plot')
par(mfrow = c(1, 1))

# Plot e.d.f. with confidence bands
#install.packages('sfsmisc')
library('sfsmisc')
ecdf.ksCI(K1, ci.col = 'black')

# Nonparametric tests for normality
#install.packages('nortest')
library(nortest)
print(cvm.test(K1))
print(cvm.test(K2))
print(ad.test(K1))
print(ad.test(K2))



# Turn off device driver
#dev.off()
