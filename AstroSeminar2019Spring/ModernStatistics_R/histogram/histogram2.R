#!/usr/bin/Rscript

# Author : Bhishan Poudel
# Date   :
# Program:

# Set the working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

BMI <- rnorm(n = 1000, m = 24.2, sd = 2.2)
hist(BMI)

histinfo <- hist(BMI)
print(histinfo)

hist(BMI, breaks = 20, main = 'Breaks=20')
hist(BMI, breaks = 5, main = 'Breaks=5')

hist(BMI, breaks = c(17, 20, 23, 26, 29, 32), main = 'Breaks is vector of breakpoints')

hist(BMI, freq = FALSE, main = 'Density plot')

hist(
    BMI,
    freq = FALSE,
    xlab = 'Body Mass Index',
    main = 'Distribution of Body Mass Index',
    col = 'lightgreen',
    xlim = c(15, 35),
    ylim = c(0, .20)
)

curve(
    dnorm(x, mean = mean(BMI), sd = sd(BMI)),
    add = TRUE,
    col = 'darkblue',
    lwd = 2
) 