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
import scipy.linalg as linalg
import math as m
from generate_data import read_data
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
from matplotlib.pyplot import title as pytitle
from matplotlib.patches import Ellipse
import bovy_plot as plot

def ex3(exclude=sc.array([1,2,3,4]),plotfilename='ex3.png', bovyprintargs={}):
    """ex3: solve exercise 3

    Input:
       exclude       - ID numbers to exclude from the analysis
       plotfilename  - filename for the output plot
    Output:
       plot
    History:
       2009-05-27 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    A= sc.ones((nsample,3))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
            pass
        else:
            Y[jj]= data[ii][1][1]
            A[jj,1]= data[ii][1][0]
            A[jj,2]= data[ii][1][0]**2.
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

    #Now plot the solution
    plot.bovy_print(**bovyprintargs)
    #plot bestfit
    xrange=[0,300]
    yrange=[0,700]
    nsamples= 1001
    xs= sc.linspace(xrange[0],xrange[1],nsamples)
    ys= sc.zeros(nsamples)
    for ii in range(nsamples):
        ys[ii]= bestfit[0]+bestfit[1]*xs[ii]+bestfit[2]*xs[ii]**2.
    plot.bovy_plot(xs,ys,'k-',xrange=xrange,yrange=yrange,
                   xlabel=r'$x$',ylabel=r'$y$',zorder=2)
    #Plot data
    errorbar(A[:,1],Y,yerr,marker='o',color='k',linestyle='None',zorder=1)
    #Put in a label with the best fit
    text(5,30,r'$y = ('+'%4.4f \pm %4.4f)\,x^2 + ( %4.2f \pm %4.2f )\,x+ ( %4.0f\pm %4.0f' % (bestfit[2], m.sqrt(bestfitvar[2,2]),bestfit[1], m.sqrt(bestfitvar[1,1]), bestfit[0],m.sqrt(bestfitvar[0,0]))+r')$')
    plot.bovy_end_print(plotfilename)
    
    return 0

            
