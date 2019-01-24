#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 12, 2016
# ref: http://harding.edu/fmccown/r/
# clear; Rscript autosBarplot.r ; rm *~

# inputs: data/autos.dat
# outputs: figures/autosBarplot.eps,autosBarplot2.eps

# set the working directory
setwd("~/Copy/Programming/R/rprograms/plotting/2dplots/")

# Read values from tab-delimited autos.dat 
autos_data <- read.table("data/autos.dat", header=T, sep="\t")

# Start device driver to save output to figure.pdf
#postscript(file="figures/autosBarplot.eps", height=3.5, width=5)

# simple plot 1
# Graph cars with specified labels for axes.  Use blue 
# borders and diagnal lines in bars.
barplot(autos_data$cars, main="Cars", xlab="Days",  
   ylab="Total", names.arg=c("Mon","Tue","Wed","Thu","Fri"), 
   border="blue", density=c(10,20,30,40,50))

# another plot 
   
# Graph autos with adjacent bars using rainbow colors
barplot(as.matrix(autos_data), main="Autos", ylab= "Total",
   beside=TRUE, col=rainbow(5))

# Place the legend at the top-left corner with no frame  
# using rainbow colors
legend("topleft", c("Mon","Tue","Wed","Thu","Fri"), cex=0.6,
   bty="n", fill=rainbow(5));
   
   
# Turn off device driver (to flush output)
#dev.off()

#####################################################################################
# a stacked bar chart and place the legend outside of the plot area:
# Read values from tab-delimited autos.dat
autos_data <- read.table("data/autos.dat", header=T, sep="\t") 

# Start device driver to save output to figure.pdf
#postscript(file="figures/autosBarplot2.eps", height=3.5, width=5)

# Expand right side of clipping rect to make room for the legend
par(xpd=T, mar=par()$mar+c(0,0,0,4))

# Graph autos (transposing the matrix) using heat colors,  
# put 10% of the space between each bar, and make labels  
# smaller with horizontal y-axis labels
barplot(t(autos_data), main="Autos", ylab="Total", 
   col=heat.colors(3), space=0.1, cex.axis=0.8, las=1,
   names.arg=c("Mon","Tue","Wed","Thu","Fri"), cex=0.8) 
   
# Place the legend at (6,30) using heat colors
legend(6, 30, names(autos_data), cex=0.8, fill=heat.colors(3));

# Turn off device driver (to flush output)
#dev.off()
  
# Restore default clipping rect
par(mar=c(5, 4, 4, 2) + 0.1)
