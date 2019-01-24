#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016

# set up a device driver to plot
#postscript(file='~/Copy/Programming/R/rprograms/plotting/legends/greek1.eps')


plot(1:3, 
     #ylab = expression("Diameter of apeture (" * mu ~ "m)"),
     ylab = expression(paste("Diameter of aperture ( ", hat(mu), " )")),
     xlab = expression("Force spaces with ~" ~ mu ~ pi * sigma ~ pi),
     main = expression("This is another Greek character with space" ~ sigma))

#example 2
# mu <- 2.8
# plot(1:3, main=bquote(mu == .(mu)))

#example 3


# turn off device driver
#dev.off()