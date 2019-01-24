#! /usr/bin/env Rscript

# Bhishan Poudel
# Jan 16, 2016

x <- seq(0,pi,0.1)
y1 <- cos(x)

# set up a device driver to plot
png(file='~/Copy/Programming/R/rprograms/plotting/multiplePlotting/multiPlot1.png')
plot(x,
     y1,
     main="Example of multiplot",
     sub=expression("x"[2]),                   # subscript
     xlab = expression(paste("Greek letter ", phi)),   # Greek Letters
     ylab = expression(paste("Greek letter ",mu)),
     col.lab="blue",                                   # color of labels
     cex.lab=0.75,                                     # size of labels relative to default
     type="l",
     col = "red")

# additional plots
y2 <- sin(x)
y3 <- tan(x)
y4 <- log(x)

lines(x,y2,col="green", lty = 2, lwd = 3)
lines(x,y2,col="green", lty = 5, lwd = 2, pch = 2)
lines(x,y3,col="black", lty = 3, lwd = 5, pch = 3, lend = 0, ljoin = 2)
lines(x,y4,col="blue", lty = 1, lwd = 2, pch = 3, lend = 2, ljoin = 1, lmitre = 2)




# turn off device driver
dev.off()
