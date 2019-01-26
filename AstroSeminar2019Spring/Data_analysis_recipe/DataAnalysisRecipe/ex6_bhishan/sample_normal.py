import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg
import math as m

def sample_normal(mean,covar,nsamples=1):
    """sample_normal: Sample a d-dimensional Gaussian distribution with
    mean and covar.

    Input:
       mean     - the mean of the Gaussian
       covar    - the covariance of the Gaussian
       nsamples - (optional) the number of samples desired

    Output:
       samples; if nsamples != 1 then a list is returned

    Dependencies:
       scipy
       scipy.stats.norm
       scipy.linalg.cholesky

    History:
       2009-05-20 - Written - Bovy (NYU)
    """
    p= covar.shape[0]
    #First lower Cholesky of covar
    L= linalg.cholesky(covar,lower=True)
    if nsamples > 1:
        out= []
    for kk in range(nsamples):
        #Generate a vector in which the elements ~N(0,1)
        y= sc.zeros(p)
        for ii in range(p):
            y[ii]= stats.norm.rvs()
        #Form the sample as Ly+mean
        thissample= sc.dot(L,y)+mean
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out