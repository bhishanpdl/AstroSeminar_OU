#############################################################################
#Copyright (c) 2010, Jo Bovy, David W. Hogg, Dustin Lang
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import scipy as sc
from scipy import special
import math as m
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.stats as stats
from generate_data import read_data
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
import bovy_plot as plot

def ex10(exclude=sc.array([1,2,3,4]),
		 plotfilenameA='ex10a.png',
		 plotfilenameB='ex10b.png',
         nburn=1000,nsamples=200000,
         parsigma=[5,.075,0.1],
		 bovyprintargs={}):
    """ex10: solve exercise 10 using MCMC sampling
    Input:
       exclude        - ID numbers to exclude from the analysis (can be None)
       plotfilename   - filename for the output plot
       nburn          - number of burn-in samples
       nsamples       - number of samples to take after burn-in
       parsigma       - proposal distribution width (Gaussian)
    Output:
       plot
    History:
       2010-05-07 - Written - Bovy (NYU)
    """
    sc.random.seed(-1) #In the interest of reproducibility (if that's a word)
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    if not exclude == None:
        nsample= ndata- len(exclude)
    else:
        nsample= ndata
    #First find the chi-squared solution, which we will use as an
    #initial guess
    #Put the data in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if not exclude == None and sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            X[jj]= data[ii][1][0]
            A[jj,1]= data[ii][1][0]
            C[jj,jj]= data[ii][2]**2.
            yerr[jj]= data[ii][2]
            jj= jj+1
    #Now compute the best fit and the uncertainties
    bestfit= sc.dot(linalg.inv(C),Y.T)
    bestfit= sc.dot(A.T,bestfit)
    bestfitvar= sc.dot(linalg.inv(C),A)
    bestfitvar= sc.dot(A.T,bestfitvar)
    bestfitvar= linalg.inv(bestfitvar)
    bestfit= sc.dot(bestfitvar,bestfit)
    initialguess= sc.array([bestfit[0],bestfit[1],0.])#(m,b,logS)
    #With this initial guess start off the sampling procedure
    initialX= objective(initialguess,X,Y,yerr)
    currentX= initialX
    bestX= initialX
    bestfit= initialguess
    currentguess= initialguess
    naccept= 0
    samples= []
    samples.append(currentguess)
    for jj in range(nburn+nsamples):
        #Draw a sample from the proposal distribution
        newsample= sc.zeros(3)
        newsample[0]= currentguess[0]+stats.norm.rvs()*parsigma[0]
        newsample[1]= currentguess[1]+stats.norm.rvs()*parsigma[1]
        newsample[2]= currentguess[2]+stats.norm.rvs()*parsigma[2]
        #Calculate the objective function for the newsample
        newX= objective(newsample,X,Y,yerr)
        #Accept or reject
        #Reject with the appropriate probability
        u= stats.uniform.rvs()
        accept=False
        try:
            test= m.exp(newX-currentX)
            if u < test:
                accept= True
        except OverflowError:
            accept= True
        if accept:
            #Accept
            currentX= newX
            currentguess= newsample
            naccept= naccept+1
        if currentX > bestX:
            bestfit= currentguess
            bestX= currentX
        samples.append(currentguess)
    if double(naccept)/(nburn+nsamples) < .5 or double(naccept)/(nburn+nsamples) > .8:
        print "Acceptance ratio was "+str(double(naccept)/(nburn+nsamples))

    samples= sc.array(samples).T[:,nburn:-1]
    print "Best-fit, overall"
    print bestfit, sc.mean(samples[2,:]), sc.median(samples[2,:])

    histmb,edges= sc.histogramdd(samples.T[:,0:2],bins=round(sc.sqrt(nsamples)/2.))
    indxi= sc.argmax(sc.amax(histmb,axis=1))
    indxj= sc.argmax(sc.amax(histmb,axis=0))
    print "Best-fit, marginalized"
    print edges[0][indxi-1], edges[1][indxj-1]
    print edges[0][indxi], edges[1][indxj]
    print edges[0][indxi+1], edges[1][indxj+1]

    print "Best-fit for S marginalized"
    histS,edgesS= sc.histogram(samples.T[:,2],bins=round(sc.sqrt(nsamples)/2.))
    indx= sc.argmax(histS)

    #Data with MAP line and sampling
    plot.bovy_print(**bovyprintargs)
    bestb= bestfit[0]
    bestm= bestfit[1]
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(xrange,bestm*sc.array(xrange)+bestb,'k-',
                   xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    errorbar(X,Y,sc.exp(bestfit[2]/2.),
             marker='o',color='k',linestyle='None',zorder=1)
    plot.bovy_text(r'$\mathrm{MAP}\ :\ y = %4.2f \,x+ %4.0f' % (bestfit[1], bestfit[0])+r'$'+'\n'+r'$\mathrm{MAP}\ :\ \sqrt{S} = %3.1f$' % (sc.exp(bestfit[2]/2.)),
                   bottom_right=True)
    plot.bovy_end_print(plotfilenameA)
    
    #Data with MAP line and sampling
    plot.bovy_print(**bovyprintargs)
    bestb= edges[0][indxi]
    bestm= edges[1][indxj]
    bestS= edgesS[indx]
    xrange=[0,300]
    yrange=[0,700]
    plot.bovy_plot(xrange,bestm*sc.array(xrange)+bestb,'k-',
                   xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    errorbar(X,Y,sc.exp(bestS/2.),
             marker='o',color='k',linestyle='None',zorder=1)
    plot.bovy_text(r'$\mathrm{marginalized\ over\ S}\ :\ y = %4.2f \,x+ %4.0f' % (bestm, bestb)+r'$'+'\n'+r'$\mathrm{marginalized\ over}\ (m,b)\ :\ \sqrt{S} = %3.1f$' % (sc.exp(bestS/2.)),
                   bottom_right=True)
    plot.bovy_end_print(plotfilenameB)

    return


def objective(pars,X,Y,yerr):
    """The objective function"""
    b= pars[0]
    s= pars[1]
    S= sc.exp(pars[2])
    return -0.5*sc.sum((Y-s*X-b)**2./S+pars[2])

