#!/usr/bin/env Rscript

# Author  : Bhishan Poudel
# Date    : Jan 8, 2016
# Program : 

# Setting working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

data <- c(5.28, 14.64, 37.25, 78.9, 44.92, 8.96, 19.22, 34.81, 
          33.89, 24.28, 6.5, 4.32, 2.77, 17.6, 33.26, 52.78, 
          5.98, 22.48, 20.11, 65.74, 35.73, 56.95, 30.61, 29.82)

hist(
    data,
    breaks = seq(0, 80, l = 6),
    freq = FALSE,
    col = "orange",
    main = "Histogram",
    xlab = "x",
    ylab = "f(x)",
    yaxs = "i",
    xaxs = "i"
)

# method 2
plot(cut(data, breaks = 4))

# method 3
number_of_bins = 4
hist(
    data,
    breaks = seq(min(data), max(data), l = number_of_bins + 1),
    freq = FALSE,
    col = "orange",
    main = "Histogram",
    xlab = "x",
    ylab = "f(x)",
    yaxs = "i",
    xaxs = "i"
)

# method 4 using ggplot2


library(ggplot2)
data <- data.frame(x = data)

ggplot(data, aes(x = x)) +
    geom_histogram(binwidth = 18,
                   color = "green",
                   fill = "grey") +
    scale_x_continuous(breaks = c(0, 20, 40, 60, 80))
                       
                       