#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016


### Change spacing to allow for multiline title
par(oma=c(2, 0, 4, 0))
### Some greek letters
mu <- 0
alpha <- 1
beta <- 2
delta <- 3
lambda <- 4
### Some graph or other
plot(1:10,1:10)
mtext(expression(bold("Meaningless Title")),
      line=3.5,cex=1.15)
mtext(bquote(paste(lambda==.(lambda),", ",
                   alpha==.(alpha),", ",
                   beta==.(beta),", ",
                   delta==.(delta),", ",
                   mu==.(mu),sep="")),
      line=2.25,cex=1.15)
