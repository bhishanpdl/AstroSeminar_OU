#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016

# ref: http://www.ats.ucla.edu/stat/r/pages/greek_letters.htm
par(mfrow = c(1, 2))
curve(dnorm, from = -3, to = 3, n = 1000, main = "Normal Probability Density Function")
text(-2, 0.3, expression(f(x) == paste(frac(1, sqrt(2 * pi * sigma^2)),
                                       " ", e^{
                                         frac(-(x - mu)^2, 2 * sigma^2)
                                       })), cex = 1.2)
x <- dnorm(seq(-3, 3, 0.001))
plot(seq(-3, 3, 0.001), cumsum(x)/sum(x), type = "l", col = "blue",
     xlab = "x", main = "Normal Cumulative Distribution Function")
text(-1.5, 0.7, expression(phi(x) == paste(frac(1, sqrt(2 * pi)),
     " ", integral(e^(-t^2/2) * dt, -infinity, x))), cex = 1.2)