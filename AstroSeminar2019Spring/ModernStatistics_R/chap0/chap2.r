#!/usr/bin/Rscript

# Bhishan Poudel
# Jan 12, 2016

# Chapter. 2 Probability
################################################################################
# set the working directory
setwd ("~/Copy/2016Spring/RProgramming/presentation/")


# Start device driver to save output
postscript(file="figures/chap2.eps", height=14, width=8)

# Set up 6 panel figure
par(mfrow=c(3,2))  # par is parameter

# Expand right side of clipping rect to make room for the legend
#par(xpd=T, mar=par()$mar+c(0,0,0,4))


# Plot upper left panel with three illustrative 
# exponential p.d.f. distributions (dexp)
xdens <- seq(0,5,0.02)
plot(xdens,dexp(xdens,rate=0.5), type='l', ylim=c(0,1.5), xlab='', ylab='Exponential p.d.f.',lty=1)

lines(xdens,dexp(xdens,rate=1), type='l', lty=2)
lines(xdens,dexp(xdens,rate=1.5), type='l', lty=3)
legend(2, 1.45, lty=1, substitute(lambda==0.5), box.lty=0)
legend(2, 1.30, lty=2, substitute(lambda==1.0), box.lty=0)
legend(2, 1.15, lty=3, substitute(lambda==1.5), box.lty=0)

# Help files to learn these function
# help(seq) ; help(plot); help(par) ; help(lines); help(legend)


# Plot upper right panel with three illustrative exponential c.d.f. distributions
plot(xdens, pexp(xdens,rate=0.5), type='l', ylim=c(0,1.0), xlab='', ylab='Exponential c.d.f.', lty=1)
lines(xdens, pexp(xdens,rate=1), type='l', lty=2)
lines(xdens, pexp(xdens,rate=1.5),type='l',lty=3)
legend(3, 0.50, lty=1, substitute(lambda==0.5), box.lty=0)
legend(3, 0.38, lty=2, substitute(lambda==1.0), box.lty=0)
legend(3, 0.26, lty=3, substitute(lambda==1.5), box.lty=0)


# Plot middle panels with illustrative normal p.d.f. and c.d.f. 
xdens <- seq(-5, 5, 0.02)
ylabdnorm <- expression(phi[mu~sigma^2] (x))
plot(xdens, dnorm(xdens, sd=sqrt(0.2)), type='l', ylim=c(0,1.0), xlab='', ylab=ylabdnorm,lty=1)
lines(xdens,dnorm(xdens, sd=sqrt(1.0)), type='l', lty=2)
lines(xdens,dnorm(xdens, sd=sqrt(5.0)), type='l', lty=3)
lines(xdens,dnorm(xdens, mean=-2.0, sd=sqrt(0.5)), type='l', lty=4)
leg1 <- expression(mu^' '==0, mu^' '==0, mu^' '==0, mu^' '==-2)
leg2 <- expression(sigma^2==0.2, sigma^2==1.0, sigma^2==5.0, sigma^2==0.5,)
legend(0.5, 1.0, lty=1:4, leg1, lwd=2, box.lty=0)
legend(3.0, 1.01, leg2, box.lty=0)
ylabpnorm <- expression(Phi[mu~sigma^2] (x))
plot(xdens,pnorm(xdens,sd=sqrt(0.2)), type='l', ylim=c(0,1.0), xlab='',ylab=ylabpnorm,lty=1)
lines(xdens,pnorm(xdens, sd=sqrt(1.0)), type='l', lty=2)
lines(xdens,pnorm(xdens, sd=sqrt(5.0)), type='l', lty=3)
lines(xdens,pnorm(xdens, mean=-2.0, sd=sqrt(0.5)), type='l', lty=4)
leg1 <- expression(mu^' '==0, mu^' '==0, mu^' '==0, mu^' '==-2)
leg2 <- expression(sigma^2==0.2, sigma^2==1.0, sigma^2==5.0, sigma^2==0.5,)
legend(0.5, 0.6, lty=1:4, leg1, lwd=2, box.lty=0)
legend(3.0, 0.61, leg2, box.lty=0)


# Plot bottom panels with illustrative lognormal p.d.f. and c.d.f. 
xdens <- seq(0,3, 0.02)
plot(xdens, dlnorm(xdens, meanlog=0, sdlog=5), type='l', ylim=c(0,2), xlab='', ylab='Lognormal density', lty=1)
lines(xdens, dlnorm(xdens, meanlog=0, sdlog=1), type='l', lty=2)
lines(xdens, dlnorm(xdens, meanlog=0, sdlog=1/2), type='l', lty=3)
lines(xdens, dlnorm(xdens, meanlog=0, sdlog=1/8), type='l', lty=4)
leg1 <- expression(sigma==5, sigma==1, sigma==1/2, sigma==1/8)
legend(1.8,1.8,lty=1:4,leg1,box.lty=0)
plot(xdens,plnorm(xdens,meanlog=0,sdlog=5),type='l',ylim=c(0,1),xlab='x', ylab='Lognormal distribution',lty=1)
lines(xdens,plnorm(xdens,meanlog=0,sdlog=1),type='l',lty=2)
lines(xdens,plnorm(xdens,meanlog=0,sdlog=1/2),type='l',lty=3)
lines(xdens,plnorm(xdens,meanlog=0,sdlog=1/8),type='l',lty=4)
leg1 <- expression(sigma==5,sigma==1,sigma==1/2,sigma==1/8)
legend(1.5,0.6,lty=1:4,leg1,box.lty=0)

# Turn off device driver (to flush output to png)
dev.off()

# Return plot to single-panel format
par(mfrow=c(1,1))

# Restore default clipping rect
par(mar=c(5, 4, 4, 2) + 0.1)

#‘mfcol, mfrow’ A vector of the form ‘c(nr, nc)’.  Subsequent
#          figures will be drawn in an ‘nr’-by-‘nc’ array on the device
#          by _columns_ (‘mfcol’), or _rows_ (‘mfrow’), respectively.

#          In a layout with exactly two rows and columns the base value
#          of ‘"cex"’ is reduced by a factor of 0.83: if there are three
#          or more of either rows or columns, the reduction factor is
#          0.66.

#          Setting a layout resets the base value of ‘cex’ and that of
#          ‘mex’ to ‘1’.

#          If either of these is queried it will give the current
#          layout, so querying cannot tell you the order in which the
#          array will be filled.



