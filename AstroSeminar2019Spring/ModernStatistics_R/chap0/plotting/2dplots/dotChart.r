#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 12, 2016
# ref: http://harding.edu/fmccown/r/
# clear; Rscript dotChart.r ; rm *~

# inputs: data/autos.dat
# outputs: figures/dotChart.eps

# set the working directory
setwd("~/Copy/Programming/R/rprograms/plotting/2dplots/")

# Start device driver to save output to figure.pdf
#postscript(file="figures/dotChart.eps", height=10.5, width=5)

# Read values from tab-delimited autos.dat
autos_data <- read.table("data/autos.dat", header=T, sep="\t")

# Create a colored dotchart for autos with smaller labels
dotchart(t(autos_data), 
         color=c("red","blue","darkgreen"),
         main="Dotchart for Autos", cex=0.8)
   
# Turn off device driver (to flush output)
#dev.off()
