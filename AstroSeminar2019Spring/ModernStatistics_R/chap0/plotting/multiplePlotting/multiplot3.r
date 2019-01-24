#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 16, 2016


number_vals<-100
x<-rnorm(number_vals)
y<-rnorm(number_vals)
plot(1, 1,
     xlim= c(min(x), max(x)),
     ylim= c(min(y), max(y)), col=0)
points(x, y)
the_expression<-paste("Plotting y vs. x",                     
                      "\n sigma x = ", format(sd(x), digits=4, scientific=F),                      
                      "\n sigma y = ", format(sd(y), digits=4, scientific=F),  sep="")
title(the_expression)
grid()