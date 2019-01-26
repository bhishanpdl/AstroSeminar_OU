import scipy as sc
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

def ex6a(exclude=sc.array([1,2,3,4]),plotfilename='ex6a.png'):
    """ex6a: solve exercise 6 by optimization of the objective function
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2009-06-01 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
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
    #Now optimize the bi-exponential objective function
    bestfitbiexp1= optimize.fmin(logbiexp,bestfit,(X,Y,yerr),disp=False)
    #Restart the optimization once using a different method
    bestfitbiexp= optimize.fmin_powell(logbiexp,bestfitbiexp1,(X,Y,yerr),disp=False)
    if linalg.norm(bestfitbiexp-bestfitbiexp1) > 10**-12:
        if linalg.norm(bestfitbiexp-bestfitbiexp1) < 10**-6:
            print("Different optimizers give slightly different results...")
        else:
            print("Different optimizers give rather different results...")
        print("The norm of the results differs by %g" % linalg.norm(bestfitbiexp-bestfitbiexp1))

    #Calculate X
    XX= 0.
    for jj in range(nsample):
        XX= XX+m.fabs(Y[jj]-bestfitbiexp[1]*X[jj]-bestfitbiexp[0])/yerr[jj]
    
    #Now plot the solution
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              #'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    #Plot data
    errorbar(X,Y,yerr,color='k',marker='o',linestyle='None')
    xlabel(r'$x$')
    ylabel(r'$y$')
    #Plot the best fit line
    xlim(0,300)
    ylim(0,700)
    xmin, xmax= xlim()
    nsamples= 1001
    xs= sc.linspace(xmin,xmax,nsamples)
    ys= sc.zeros(nsamples)
    for ii in range(nsamples):
        ys[ii]= bestfitbiexp[0]+bestfitbiexp[1]*xs[ii]
    if bestfitbiexp[0] < 0:
        sgn_str= '-'
    else:
        sgn_str= '+'
    label= r'$y = %4.2f\, x'% (bestfitbiexp[1]) +sgn_str+ '%4.0f ' % m.fabs(bestfitbiexp[0])+r'; X = '+ '%3.1f' % XX+'$'
    plot(xs,ys,color='k',ls='--',label=label)
    l=legend(loc=(.3,.1),numpoints=8)
    l.draw_frame(False)
    plot(xs,ys,'k--')
    xlim(0,300)
    ylim(0,700)

    print('Creating: ', plotfilename)
    savefig(plotfilename,format='png')
    
    return 0


def logbiexp(mb,X,Y,yerr):
    """logbiexp: evaluates the logarithm of the objective function
    Input:
       mb=(b,m)   - as in y=mx+b
       X       - independent variable
       Y       - dependent variable
       yerr    - error on the Y
    History:
       2009-06-01 - Written - Bovy (NYU)
    """
    out= 0.
    for ii in range(len(X)):
        out= out+ m.fabs(Y[ii]-mb[1]*X[ii]-mb[0])/yerr[ii]
    return out


def ex6b(exclude=sc.array([1,2,3,4]),plotfilename='ex6b.png'):
    """ex6b: solve exercise 6 using a simulated annealing optimization
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
    Output:
       plot
    History:
       2009-06-02 - Written - Bovy (NYU)
    """
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the dat in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
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
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #With this initial guess start off the annealing procedure
    initialchisq= nsample*10.
    chisq= initialchisq
    bestfit= initialguess
    nonglobal= True
    print("Performing 10 runs of the simulating basinhopping optimization algorithm")
    for jj in range(10):#Do ten runs of the sa algorithm
        sc.random.seed(jj+1) #In the interest of reproducibility (if that's a word)
        minimizer_kwargs = {"args": (X,Y,yerr)}
        bestfitbiexp= optimize.basinhopping(logbiexp,x0=initialguess,minimizer_kwargs=minimizer_kwargs,niter=100)

    # print(bestfitbiexp.keys()) # dict_keys(['lowest_optimization_result', 
    # # 'message', 'minimization_failures', 'nit', 'x', 'nfev', 'njev', 'fun'])
    # print(bestfitbiexp.x, bestfitbiexp.fun)
    # print(chisq)
    # print(bestfit)
    print(bestfitbiexp)
    # NOTE:  result of anneal (not basinhopping) res[0] is obtained min
    #        and res[1] is function value at that minimum.
    #        but result of basinhopping is OpitimizeResult object
    #        with attributes .x and .fun with others.
    #
    #        res[0] ==> res.x     ndarray
    #        res[1] ==> res.fun   function value at ndarray
    #        res[6] ==> res.status  success(bool) status(int)
    #
    if bestfitbiexp.fun < chisq:
        bestfit= bestfitbiexp.x
        chisq= bestfitbiexp.fun

    bestfitsbiexp= bestfit

    #Now plot the solution
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              #'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    #Plot data
    errorbar(X,Y,yerr,color='k',marker='o',linestyle='None')
    xlabel(r'$x$')
    ylabel(r'$y$')
    xlim(0,300)
    ylim(0,700)
    xmin, xmax= xlim()
    nsamples= 1001
    xs= sc.linspace(xmin,xmax,nsamples)
    ys= sc.zeros(nsamples)
    for ii in range(nsamples):
        ys[ii]= bestfitsbiexp[0]+bestfitsbiexp[1]*xs[ii]
    if bestfitsbiexp[0] < 0:
        sgn_str= '-'
    else:
        sgn_str= '+'
    label= r'$y = %4.2f\, x'% (bestfitsbiexp[1]) +sgn_str+ '%4.0f ' % m.fabs(bestfitsbiexp[0])+r'; X = '+ '%3.1f' % chisq+'$'
    plot(xs,ys,color='k',ls='--',label=label)
    l=legend(loc=(.3,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)

    print('Creating: ', plotfilename)
    savefig(plotfilename,format='png')

    return 0



def ex6c(exclude=sc.array([1,2,3,4]),plotfilename='ex6c.png',nburn=100,nsamples=10000,parsigma=[5,.075]):
    """ex6c: solve exercise 6 using MCMC sampling
    Input:
       exclude        - ID numbers to exclude from the analysis
       plotfilename   - filename for the output plot
       nburn          - number of burn-in samples
       nsamples       - number of samples to take after burn-in
       parsigma       - proposal distribution width (Gaussian)
    Output:
       plot
    History:
       2009-06-02 - Written - Bovy (NYU)
    """
    sc.random.seed(100) #In the interest of reproducibility (if that's a word)
    #Read the data
    data= read_data('data_yerr.dat')
    ndata= len(data)
    nsample= ndata- len(exclude)
    #First find the chi-squared solution, which we will use as an
    #initial gues for the bi-exponential optimization
    #Put the data in the appropriate arrays and matrices
    Y= sc.zeros(nsample)
    X= sc.zeros(nsample)
    A= sc.ones((nsample,2))
    C= sc.zeros((nsample,nsample))
    yerr= sc.zeros(nsample)
    jj= 0
    for ii in range(ndata):
        if sc.any(exclude == data[ii][0]):
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
    initialguess= sc.array([bestfit[0],bestfit[1]])
    #With this initial guess start off the sampling procedure
    initialX= 0.
    for jj in range(nsample):
        initialX= initialX+m.fabs(Y[jj]-bestfit[1]*X[jj]-bestfit[0])/yerr[jj]
    currentX= initialX
    bestX= initialX
    bestfit= initialguess
    currentguess= initialguess
    naccept= 0
    for jj in range(nburn+nsamples):
        #Draw a sample from the proposal distribution
        newsample= sc.zeros(2)
        newsample[0]= currentguess[0]+stats.norm.rvs()*parsigma[0]
        newsample[1]= currentguess[1]+stats.norm.rvs()*parsigma[1]
        #Calculate the objective function for the newsample
        newX= logbiexp(newsample,X,Y,yerr)
        #Accept or reject
        #Reject with the appropriate probability
        u= stats.uniform.rvs()
        if u < m.exp(currentX-newX):
            #Accept
            currentX= newX
            currentguess= newsample
            naccept= naccept+1
        if currentX < bestX:
            bestfit= currentguess
            bestX= currentX
    bestfitsbiexp= bestfit
    if double(naccept)/(nburn+nsamples) < .5 or double(naccept)/(nburn+nsamples) > .8:
        print("Acceptance ratio was "+str(double(naccept)/(nburn+nsamples)))

    #Now plot the solution
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': 12,
              #'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size}
    rcParams.update(params)
    #Plot data
    errorbar(X,Y,yerr,color='k',marker='o',linestyle='None')
    xlabel(r'$x$')
    ylabel(r'$y$')
    xlim(0,300)
    ylim(0,700)
    xmin, xmax= xlim()
    nsamples= 1001
    xs= sc.linspace(xmin,xmax,nsamples)
    ys= sc.zeros(nsamples)
    for ii in range(nsamples):
        ys[ii]= bestfitsbiexp[0]+bestfitsbiexp[1]*xs[ii]
    if bestfitsbiexp[0] < 0:
        sgn_str= '-'
    else:
        sgn_str= '+'
    label= r'$y = %4.2f\, x'% (bestfitsbiexp[1]) +sgn_str+ '%4.0f ' % m.fabs(bestfitsbiexp[0])+r'; X = '+ '%3.1f' % bestX+'$'
    plot(xs,ys,color='k',ls='--',label=label)
    l=legend(loc=(.3,.1),numpoints=8)
    l.draw_frame(False)
    xlim(0,300)
    ylim(0,700)
    print('Creating: ', plotfilename)
    savefig(plotfilename,format='png')
    
    return 0

if __name__ == '__main__':
    # run the program
    # ex6a()
    # ex6b()
    ex6c()