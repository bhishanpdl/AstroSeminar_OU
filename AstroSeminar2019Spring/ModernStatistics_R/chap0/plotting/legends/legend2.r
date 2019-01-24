#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016

# set up a device driver to plot
#postscript(file='~/Copy/Programming/R/rprograms/plotting/legends/legend2.eps')

## Using 'ncol' :
x <- 0:64/64
matplot(x, outer(x, 1:7, function(x, k) sin(k * pi * x)),
        type = "o", col = 1:7, ylim = c(-1, 1.5), pch = "*")
op <- par(bg = "antiquewhite1")
legend(0, 1.5, paste("sin(", 1:7, "pi * x)"), col = 1:7, lty = 1:7,
       pch = "*", ncol = 4, cex = 0.8)
# legend(.8,1.2, paste("sin(", 1:7, "pi * x)"), col = 1:7, lty = 1:7,
#        pch = "*", cex = 0.8)
# legend(0, -.1, paste("sin(", 1:4, "pi * x)"), col = 1:4, lty = 1:4,
#        ncol = 2, cex = 0.8)
# legend(0, -.4, paste("sin(", 5:7, "pi * x)"), col = 4:6,  pch = 24,
#        ncol = 2, cex = 1.5, lwd = 2, pt.bg = "pink", pt.cex = 1:3)
par(op)

# turn off device driver
#dev.off()