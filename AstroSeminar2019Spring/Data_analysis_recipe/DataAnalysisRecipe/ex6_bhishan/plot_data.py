# %load bovy_plot.py
##############################################################################
#
#   bovy_plot.py: general wrappers for matplotlib plotting
#
#       'public' methods:
#                         bovy_end_print
#                         bovy_dens2d
#                         bovy_hist
#                         bovy_plot
#                         bovy_print
#                         scatterplot (like hogg_scatterplot)
#                         bovy_text
#
#############################################################################
import re
import math as m
import scipy as sc
from scipy import special
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import NullFormatter
_DEFAULTNCNTR= 10

def bovy_end_print(filename,**kwargs):
    """
    NAME:
       bovy_end_print

    PURPOSE:
       saves the current figure(s) to filename

    INPUT:
       filename - filename for plot (with extension)

    OPTIONAL INPUTS:
       format - file-format

    OUTPUT:
       (none)

    HISTORY:
       2009-12-23 - Written - Bovy (NYU)
    """

    if 'format' in kwargs:
        pyplot.savefig(filename,format=kwags['format'])
    else:
        pyplot.savefig(filename,format=re.split(r'\.',filename)[-1])
    
def bovy_hist(x,xlabel=None,ylabel=None,overplot=False,**kwargs):
    """
    NAME:
       bovy_hist

    PURPOSE:
       wrapper around matplotlib's hist function

    INPUT:
       x - array to histogram
       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed
       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed
       + all pyplot.hist keywords

    OUTPUT:
       (from the matplotlib docs:
       http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.hist)

       The return value is a tuple (n, bins, patches)
       or ([n0, n1, ...], bins, [patches0, patches1,...])
       if the input contains multiple data

    HISTORY:
       2009-12-23 - Written - Bovy (NYU)
    """

    if not overplot:
        pyplot.figure()

    out= pyplot.hist(x,**kwargs)

    _add_axislabels(xlabel,ylabel)

    if not 'range' in kwargs:
        pyplot.xlim(x.min(),x.max())

    else:
        pyplot.xlim(kwargs['range'])

    _add_ticks()

    return out


def bovy_plot(*args,**kwargs):
    """
    NAME:
       bovy_plot

    PURPOSE:

       wrapper around matplotlib's plot function

    INPUT:

       see http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot

       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed

       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

       xrange

       yrange

       overplot=True does not start a new figure

    OUTPUT:

    HISTORY:

       2009-12-28 - Written - Bovy (NYU)
    """

    if 'overplot' in kwargs and kwargs['overplot']:
        kwargs.pop('overplot')
        overplot=True

    elif 'overplot' in kwargs:
        kwargs.pop('overplot')
        pyplot.figure()
        overplot=False

    else:
        pyplot.figure()
        overplot=False

    ax=pyplot.gca()
    ax.set_autoscale_on(False)

    if 'xlabel' in kwargs:
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')

    else:
        xlabel=None

    if 'ylabel' in kwargs:
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')

    else:
        ylabel=None

    if 'xrange' in kwargs:
        xlimits=kwargs['xrange']
        kwargs.pop('xrange')

    else:
        xlimits=(args[0].min(),args[0].max())

    if 'yrange' in kwargs:
        ylimits=kwargs['yrange']
        kwargs.pop('yrange')

    else:
        ylimits=(args[1].min(),args[1].max())

    out= pyplot.plot(*args,**kwargs)

    if overplot:
        pass

    else:
        pyplot.xlim(*xlimits)
        pyplot.ylim(*ylimits)

        _add_axislabels(xlabel,ylabel)
        _add_ticks()

    return out


def bovy_dens2d(X,**kwargs):
    """
    NAME:
       bovy_dens2d

    PURPOSE:
       plot a 2d density with optional contours

    INPUT:
       first argument is the density

       matplotlib.pyplot.imshow keywords (see http://matplotlib.sourceforge.net/api/axes_api.html#matplotlib.axes.Axes.imshow)
       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed
       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed
       xrange
       yrange
       noaxes - don't plot any axes
       overplot - if True, overplot

       Contours:
       contours - if True, draw contours (10 by default)
       levels - contour-levels
       cntrmass - if True, the density is a probability and the levels 
                  are probability masses contained within the contour
       cntrcolors - colors for contours (single color or array)
       cntrlabel - label the contours
       cntrlw, cntrls - linewidths and linestyles for contour
       cntrlabelsize, cntrlabelcolors,cntrinline - contour arguments

    OUTPUT:

    HISTORY:
       2010-03-09 - Written - Bovy (NYU)
    """

    if 'overplot' in kwargs:
        overplot= kwargs['overplot']
        kwargs.pop('overplot')

    else:
        overplot= False

    if not overplot:
        pyplot.figure()

    ax=pyplot.gca()
    ax.set_autoscale_on(False)

    if 'xlabel' in kwargs:
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')

    else:
        xlabel=None

    if 'ylabel' in kwargs:
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')

    else:
        ylabel=None

    if 'extent' in kwargs:
        extent= kwargs['extent']
        kwargs.pop('extent')

    else:
        if 'xrange' in kwargs:
            xlimits=list(kwargs['xrange'])
            kwargs.pop('xrange')

        else:
            xlimits=[0,X.shape[0]]

        if 'yrange' in kwargs:
            ylimits=list(kwargs['yrange'])
            kwargs.pop('yrange')

        else:
            ylimits=[0,X.shape[1]]

        extent= xlimits+ylimits

    if 'noaxes' in kwargs:
        noaxes= kwargs['noaxes']
        kwargs.pop('noaxes')

    else:
        noaxes= False

    if 'contours' in kwargs and kwargs['contours']:
        contours= True
        kwargs.pop('contours')

    elif kwargs.has_key('levels') or 'cntrmass' in kwargs:
        contours= True

    else:
        contours= False

        if 'contours' in kwargs:
            kwargs.pop('contours')

    if 'levels' in kwargs:
        levels= kwargs['levels']
        kwargs.pop('levels')

    elif contours:
        if 'cntrmass' in kwargs and kwargs['cntrmass']:
            levels= sc.linspace(0.,1.,_DEFAULTNCNTR)

        elif True in sc.isnan(sc.array(X)):
            levels= sc.linspace(sc.nanmin(X),sc.nanmax(X),_DEFAULTNCNTR)

        else:
            levels= sc.linspace(sc.amin(X),sc.amax(X),_DEFAULTNCNTR)

    if 'cntrmass' in kwargs and kwargs['cntrmass']:
        cntrmass= True
        kwargs.pop('cntrmass')

    else:
        cntrmass= False
        if 'cntrmass' in kwargs: kwargs.pop('cntrmass')

    if 'cntrcolors' in kwargs:
        cntrcolors= kwargs['cntrcolors']
        kwargs.pop('cntrcolors')

    elif contours:
        cntrcolors='k'

    if 'cntrlabel' in kwargs and kwargs['cntrlabel']:
        cntrlabel= True
        kwargs.pop('cntrlabel')

    else:
        cntrlabel= False
        if 'cntrlabel' in kwargs: kwargs.pop('cntrlabel')

    if 'cntrlw' in kwargs:
        cntrlw= kwargs['cntrlw']
        kwargs.pop('cntrlw')

    elif contours:
        cntrlw= None

    if 'cntrls' in kwargs:
        cntrls= kwargs['cntrls']
        kwargs.pop('cntrls')

    elif contours:
        cntrls= None

    if 'cntrlabelsize' in kwargs:
        cntrlabelsize= kwargs['cntrlabelsize']
        kwargs.pop('cntrlabelsize')

    elif contours:
        cntrlabelsize= None

    if 'cntrlabelcolors' in kwargs:
        cntrlabelcolors= kwargs['cntrlabelcolors']
        kwargs.pop('cntrlabelcolors')

    elif contours:
        cntrlabelcolors= None

    if 'cntrinline' in kwargs:
        cntrinline= kwargs['cntrinline']
        kwargs.pop('cntrinline')

    elif contours:
        cntrinline= None

    if 'retCumImage' in kwargs:
        retCumImage= kwargs['retCumImage']
        kwargs.pop('retCumImage')

    else:
        retCumImage= False

    out= pyplot.imshow(X,extent=extent,**kwargs)
    pyplot.axis(extent)

    _add_axislabels(xlabel,ylabel)
    _add_ticks()

    if contours:
        if 'aspect' in kwargs:
            aspect= kwargs['aspect']

        else:
            aspect= None

        if 'origin' in kwargs:
            origin= kwargs['origin']

        else:
            origin= None

        if cntrmass:
            #Sum from the top down!
            sortindx= sc.argsort(X.flatten())[::-1]
            cumul= sc.cumsum(sc.sort(X.flatten())[::-1])/sc.sum(X.flatten())
            cntrThis= sc.zeros(sc.prod(X.shape))
            cntrThis[sortindx]= cumul
            cntrThis= sc.reshape(cntrThis,X.shape)

        else:
            cntrThis= X

        cont= pyplot.contour(cntrThis,levels,colors=cntrcolors,
                             linewidths=cntrlw,extent=extent,aspect=aspect,
                             linestyles=cntrls,origin=origin)

        if cntrlabel:
            pyplot.clabel(cont,fontsize=cntrlabelsize,colors=cntrlabelcolors,
                          inline=cntrinline)

    if noaxes:
        ax.set_axis_off()

    if retCumImage:
        return cntrThis

    else:
        return out

def bovy_print(fig_width=5,fig_height=5,axes_labelsize=16,
               text_fontsize=11,legend_fontsize=12,
               xtick_labelsize=10,ytick_labelsize=10,
               xtick_minor_size=2,ytick_minor_size=2,
               xtick_major_size=4,ytick_major_size=4):
    """
    NAME:
       bovy_print

    PURPOSE:
       setup a figure for plotting

    INPUT:
       fig_width - width in inches
       fig_height - height in inches
       axes_labelsize - size of the axis-labels
       #text_fontsize - font-size of the text (if any)
       legend_fontsize - font-size of the legend (if any)
       xtick_labelsize - size of the x-axis labels
       ytick_labelsize - size of the y-axis labels
       xtick_minor_size - size of the minor x-ticks
       ytick_minor_size - size of the minor y-ticks

    OUTPUT:
       (none)

    HISTORY:
       2009-12-23 - Written - Bovy (NYU)
    """

    fig_size =  [fig_width,fig_height]
    params = {'axes.labelsize': axes_labelsize,
              #'text.fontsize': text_fontsize,
              'legend.fontsize': legend_fontsize,
              'xtick.labelsize':xtick_labelsize,
              'ytick.labelsize':ytick_labelsize,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.major.size' : xtick_major_size,
              'ytick.major.size' : ytick_major_size,
              'xtick.minor.size' : xtick_minor_size,
              'ytick.minor.size' : ytick_minor_size}

    pyplot.rcParams.update(params)

    rc('text.latex', preamble=r'\usepackage{amsmath}')


def bovy_text(*args,**kwargs):
    """
    NAME:
       bovy_text

    PURPOSE:
       thin wrapper around matplotlib's text and annotate
       use keywords:
          'bottom_left=True'
          'bottom_right=True'
          'top_left=True'
          'top_right=True'
          'title=True'

       to place the text in one of the corners or use it as the title

    INPUT:
       see matplotlib's text
          (http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.text)

    OUTPUT:
       prints text on the current figure

    HISTORY:
       2010-01-26 - Written - Bovy (NYU)
    """

    if 'title' in kwargs:
        kwargs.pop('title')
        pyplot.annotate(args[0],(0.5,1.05),xycoords='axes fraction',
                        horizontalalignment='center',
                        verticalalignment='top')

    elif 'bottom_left' in kwargs:
        kwargs.pop('bottom_left')
        pyplot.annotate(args[0],(0.05,0.05),xycoords='axes fraction')

    elif 'bottom_right' in kwargs:
        kwargs.pop('bottom_right')
        pyplot.annotate(args[0],(0.95,0.05),xycoords='axes fraction',
                        horizontalalignment='right')

    elif 'top_right' in kwargs:
        kwargs.pop('top_right')
        pyplot.annotate(args[0],(0.95,0.95),xycoords='axes fraction',
                        horizontalalignment='right',
                        verticalalignment='top')

    elif 'top_left' in kwargs:
        kwargs.pop('top_left')
        pyplot.annotate(args[0],(0.05,0.95),xycoords='axes fraction',
                        verticalalignment='top')

    else:
        pyplot.text(*args,**kwargs)

def scatterplot(x,y,*args,**kwargs):
    """
    NAME:
       scatterplot

    PURPOSE:
       make a 'smart' scatterplot that is a density plot in high-density

       regions and a regular scatterplot for outliers

    INPUT:
       x, y
       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed
       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed
       xrange
       yrange
       bins - number of bins to use in each dimension
       weights - data-weights
       aspect - aspect ratio
       onedhists - if True, make one-d histograms on the sides
       onedhistcolor, onedhistfc, onedhistec

    OUTPUT:
    HISTORY:
       2010-04-15 - Written - Bovy (NYU)

    """

    if 'xlabel' in kwargs:
        xlabel= kwargs['xlabel']
        kwargs.pop('xlabel')

    else:
        xlabel=None

    if 'ylabel' in kwargs:
        ylabel= kwargs['ylabel']
        kwargs.pop('ylabel')

    else:
        ylabel=None

    if 'xrange' in kwargs:
        xrange=kwargs['xrange']
        kwargs.pop('xrange')

    else:
        xrange=[x.min(),x.max()]
    if 'yrange' in kwargs:
        yrange=kwargs['yrange']
        kwargs.pop('yrange')

    else:
        yrange=[y.min(),y.max()]

    ndata= len(x)
    if 'bins' in kwargs:
        bins= kwargs['bins']
        kwargs.pop('bins')

    else:
        bins= round(0.3*sc.sqrt(ndata))

    if 'weights' in kwargs:
        weights= kwargs['weights']
        kwargs.pop('weights')

    else:
        weights= None

    if 'levels' in kwargs:
        levels= kwargs['levels']
        kwargs.pop('levels')

    else:
        levels= special.erf(0.5*sc.arange(1,4))

    if 'aspect' in kwargs:
        aspect= kwargs['aspect']
        kwargs.pop('aspect')

    else:
        aspect= (xrange[1]-xrange[0])/(yrange[1]-yrange[0])

    if 'onedhists' in kwargs:
        onedhists= kwargs['onedhists']
        kwargs.pop('onedhists')

    else:
        onedhists= False

    if 'onedhisttype' in kwargs:
        onedhisttype= kwargs['onedhisttype']
        kwargs.pop('onedhisttype')

    else:
        onedhisttype= 'step'

    if 'onedhistcolor' in kwargs:
        onedhistcolor= kwargs['onedhistcolor']
        kwargs.pop('onedhistcolor')

    else:
        onedhistcolor= 'k'

    if 'onedhistfc' in kwargs:
        onedhistfc=kwargs['onedhistfc']
        kwargs.pop('onedhistfc')

    else:
        onedhistfc= 'w'

    if 'onedhistec' in kwargs:
        onedhistec=kwargs['onedhistec']
        kwargs.pop('onedhistec')

    else:
        onedhistec= 'k'

    if onedhists:
        fig= pyplot.figure()
        nullfmt   = NullFormatter()         # no labels

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = pyplot.axes(rect_scatter)
        axHistx = pyplot.axes(rect_histx)
        axHisty = pyplot.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        axHisty.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        fig.sca(axScatter)

    data= sc.array([x,y]).T
    hist, edges= sc.histogramdd(data,bins=bins,range=[xrange,yrange],
                                weights=weights)

    cumimage= bovy_dens2d(hist.T,contours=True,levels=levels,cntrmass=True,
                          cntrcolors='k',cmap=cm.gist_yarg,origin='lower',
                          xrange=xrange,yrange=yrange,xlabel=xlabel,
                          ylabel=ylabel,interpolation='nearest',
                          retCumImage=True,aspect=aspect,
                          overplot=onedhists)

    binxs= []
    xedge= edges[0]
    for ii in range(len(xedge)-1):
        binxs.append((xedge[ii]+xedge[ii+1])/2.)

    binxs= sc.array(binxs)
    binys= []
    yedge= edges[1]

    for ii in range(len(yedge)-1):
        binys.append((yedge[ii]+yedge[ii+1])/2.)

    binys= sc.array(binys)
    cumInterp= interpolate.RectBivariateSpline(binxs,binys,cumimage.T,
                                               kx=1,ky=1)

    cums= []
    for ii in range(len(x)):
        cums.append(cumInterp(x[ii],y[ii])[0,0])

    cums= sc.array(cums)
    plotx= x[cums > levels[-1]]
    ploty= y[cums > levels[-1]]
    if not weights == None:
        w8= weights[cums > levels[-1]]
        for ii in range(len(plotx)):
            bovy_plot(plotx[ii],ploty[ii],overplot=True,
                      color='%.2f'%(1.-w8[ii]),*args,**kwargs)

    else:
        bovy_plot(plotx,ploty,overplot=True,*args,**kwargs)

    #Add onedhists
    if not onedhists:
        return

    axHistx.hist(x, bins=bins,normed=True,histtype=onedhisttype,range=xrange,
                 color=onedhistcolor,fc=onedhistfc,ec=onedhistec)
    axHisty.hist(y, bins=bins, orientation='horizontal',normed=True,
                 histtype=onedhisttype,range=yrange,
                 color=onedhistcolor,fc=onedhistfc,ec=onedhistec)

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )


def _add_axislabels(xlabel,ylabel):
    """
    NAME:
       _add_axislabels

    PURPOSE:
       add axis labels to the current figure

    INPUT:
       xlabel - (raw string!) x-axis label, LaTeX math mode, no $s needed
       ylabel - (raw string!) y-axis label, LaTeX math mode, no $s needed

    OUTPUT:
       (none; works on the current axes)

    HISTORY:
       2009-12-23 - Written - Bovy (NYU)
    """

    if xlabel != None:
        if xlabel[0] != '$':
            thisxlabel=r'$'+xlabel+'$'

        else:
            thisxlabel=xlabel

        pyplot.xlabel(thisxlabel)

    if ylabel != None:
        if ylabel[0] != '$':
            thisylabel=r'$'+ylabel+'$'

        else:
            thisylabel=ylabel

        pyplot.ylabel(thisylabel)


def _add_ticks():
    """
    NAME:
       _add_ticks

    PURPOSE:
       add minor axis ticks to a plot

    INPUT:
       (none; works on the current axes)

    OUTPUT:
       (none; works on the current axes)

    HISTORY:
       2009-12-23 - Written - Bovy (NYU)
    """

    ax=pyplot.gca()
    xstep= ax.xaxis.get_majorticklocs()
    xstep= xstep[1]-xstep[0]
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(xstep/5.))
    ystep= ax.yaxis.get_majorticklocs()
    ystep= ystep[1]-ystep[0]
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(ystep/5.))

# another file
# %load generate_data.py
#############################################################################
import re
import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg
import math as m
import numpy as nu
# from sample_wishart import sample_wishart
# from sample_normal import sample_normal

def generate_data(ndata=20,nback=4,yerr=.05,errprop=2,wishartshape=5):
    """generate_data: Generate the data that is to be fit with a straight line

    Input:
       ndata    - Total number of data points to generate
       nback    - Number of data points to generate from the background
       yerr     - typical fractional y error
       errprop  - proportionality constant between typical y and typical
                  x error
       wishartshape - shape parameter for the Wishart density from which
                      the error covariances are drawn

    Output:
       list of { array(data point), array(errorcovar) }
    
    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    nu.random.seed(8) #In the interest of reproducibility (if that's a word)
    #The distribution underlying the straight line is a Gaussian, with a large
    #eigenvalue in the direction of the line and a small eigenvalue in the
    #direction orthogonal to this
    #Draw a random slope (by drawing a random angle such that tan angle = slope
    alpha= stats.uniform.rvs()*m.pi-m.pi/2.
    slope= m.tan(alpha)
    #Draw a random intercept from intercept ~ 1/intercept intercept \in [.1,10]
    intercept= stats.uniform.rvs()*2.-1.
    intercept= 10.**intercept
    #print slope, intercept
    rangey= intercept*10.
    rangeline= rangey/m.sin(alpha)
    rangex= rangey/slope
    #Now draw the variances of the underlying Gaussian
    #We want one to be big
    multiplerangeline= 1.
    multiplerange2= 10.
    sigma1= nu.random.gamma(2,.5/(multiplerangeline*rangeline)**2)
    sigma1= 1/m.sqrt(sigma1)
    #And the other one the be small
    sigma2= nu.random.gamma(2,.5*(multiplerange2*rangeline)**2)
    sigma2= 1/m.sqrt(sigma2)
    covar= sc.array([[sigma1**2,0.],[0.,sigma2**2]])
    #Rotate the covariance matrix
    rotationmatrix= sc.array([[m.cos(alpha),-m.sin(alpha)],
                              [m.sin(alpha),m.cos(alpha)]])
    modelcovar= sc.dot(rotationmatrix,covar)
    modelcovar=sc.dot(modelcovar,rotationmatrix.transpose())
    #Also set the mean
    modelmean= sc.array([0.,intercept+5*rangey])
    modelmean[0]= (modelmean[1]-intercept)/slope
    #The background covar
    backcovar= sc.array([[4*rangex**2.,0.],[0.,4*rangey**2.]])
    #Now start drawing samples from this
    out=[]
    for ii in range(ndata):
        #First set-up an error covariance. Use the fractional error to
        #multiply the ymean, use the proportionality between yerr and xerr
        #to get the error in x, and draw a random angle for the correlation
        #But not allow for completely correlated erors
        #Draw a random error covariance from an inverse Wishart
        #distribution that has the constructed error covariance as its' center'
        correlation_angle= stats.uniform.rvs()*m.pi/2+m.pi/4
        thisyerr= (yerr*modelmean[1])**2.
        thisxerr= thisyerr/errprop/slope**2.
        thiscorrelation= m.cos(correlation_angle)
        thiscovxy= thiscorrelation*m.sqrt(thisxerr*thisyerr)
        thissampleerr= sc.array([[thisxerr,thiscovxy],[thiscovxy,thisyerr]])
        sampleerr= sample_wishart(
            wishartshape,linalg.inv(thissampleerr)/wishartshape)
        sampleerr= linalg.inv(sampleerr)
        #Now draw a sample from the model distribution convolved with this
        #error distribution
        if ii < nback:
            samplethiscovar= sampleerr+backcovar
        else:
            samplethiscovar= sampleerr+modelcovar
        thissample= sample_normal(modelmean,samplethiscovar)
        sample=[]
        sample.append(thissample)
        sample.append(sampleerr)
        out.append(sample)
        
    return out

def sign(x):
    if x < 0: return -1
    else: return 1
    
def write_table_to_file(filename,latex=False,allerr=False,ndec=[0,0,0,0,2]):
    """write_table_to_file: Write the generated data to a latex table
    Includes {x_i,y_i,sigma_yi}

    Input:
       filename  - filename for table
       latex     - Write latex file
       allerr    - If True, write all of the errors
       ndec      - number of decimal places (array with five members)

    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    #First generate the data
    data= generate_data()
    #Set up the file
    outfile=open(filename,'w')
    if allerr:
        ncol= 5
    else:
        ncol= 3
    #First write all of the table header
    nextra= 0
    if latex:
        outfile.write(r'\begin{deluxetable}{')
        outfile.write('r')
        for jj in range(ncol):
            outfile.write('r')
            if ndec[jj] != 0:
                nextra= nextra+1
                outfile.write(r'@{.}l')
        outfile.write('}\n')
        ntablecols= ncol+nextra+1
        outfile.write(r'\tablecolumns{'+str(ntablecols)+'}'+'\n')
        outfile.write(r'\tablehead{ID &')
        #x
        if ndec[0] != 0:
            outfile.write(r'\multicolumn{2}{c}{$x$} & ')
        else:
            outfile.write(r'$x$ & ')
        #y
        if ndec[1] != 0:
            outfile.write(r'\multicolumn{2}{c}{$y$} & ')
        else:
            outfile.write(r'$y$ & ')
        #sigma_y
        if ndec[2] != 0:
            outfile.write(r'\multicolumn{2}{c}{$\sigma_y$}')
        else:
            outfile.write(r'$\sigma_y$')
        if allerr:
            #sigma_x
            if ndec[3] != 0:
                outfile.write(r' & \multicolumn{2}{c}{$\sigma_x$} & ')
            else:
                outfile.write(r' & $\sigma_x$ & ')
            #rho_{xy}
            if ndec[4] != 0:
                outfile.write(r' \multicolumn{2}{c}{$\rho_{xy}$}')
            else:
                outfile.write(r' $\rho_{xy}')
        outfile.write(r'}'+'\n')
        outfile.write(r'\tablewidth{0pt}'+'\n')
        outfile.write(r'\startdata'+'\n')
    else:
        if allerr:
            outfile.write('#Data from Table 2\n')
            outfile.write('#ID\tx\ty\t\sigma_y\t\sigma_x\t'+r'\rho_{xy}'+'\n')
        else:
            outfile.write('#Data from Table 1\n')
            outfile.write('#ID\tx\ty\t\sigma_y\n')
    #Then write the data
    for ii in range(len(data)):
        #Write the ID
        if latex:
            outfile.write(str(ii+1)+' & ')
        else:
            outfile.write(str(ii+1)+'\t')
        #Write x and y
        for jj in range(2):
            if sign(data[ii][0][jj]) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(data[ii][0][jj]))
            dec_part= long(round(10**ndec[jj]*abs(data[ii][0][jj]-long(data[ii][0][jj]))))
            if dec_part >= 10**ndec[jj]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[jj]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[jj])
            if latex:
                if ndec[jj] != 0:
                    outfile.write(sign_str+int_part+' & '+dec_part + ' & ')
                else:
                    outfile.write(sign_str+int_part+' & ')
            else:
                if ndec[jj] != 0:
                    outfile.write(sign_str+int_part+'.'+dec_part+'\t')
                else:
                    outfile.write(sign_str+int_part+'\t')
        #Write sigma_y
        sigma_y= m.sqrt(data[ii][1][1,1])
        if sign(sigma_y) == -1:
            sign_str= '-'
        else:
            sign_str= ''
        int_part=abs(long(sigma_y))
        dec_part= long(round(10**ndec[2]*abs(sigma_y-long(sigma_y))))
        if dec_part >= 10**ndec[2]:
            int_part = int_part+1
            dec_part= dec_part-10**ndec[2]
        int_part= str(int_part)
        if dec_part == 0:
            sign_str=''
        dec_part='%i' % dec_part
        dec_part= dec_part.zfill(ndec[2])
        if latex:
            if ndec[2] != 0:
                outfile.write(sign_str+int_part+' & '+dec_part)
            else:
                outfile.write(sign_str+int_part)
        else:
            if ndec[2] != 0:
                outfile.write(sign_str+int_part+'.'+dec_part)
            else:
                outfile.write(sign_str+int_part)
        if allerr:
            #Write sigma_x
            sigma_x= m.sqrt(data[ii][1][0,0])
            if sign(sigma_x) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(sigma_x))
            dec_part= long(round(10**ndec[3]*abs(sigma_x-long(sigma_x))))
            if dec_part >= 10**ndec[3]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[3]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[3])
            if latex:
                if ndec[3] != 0:
                    outfile.write(' & '+sign_str+int_part+' & '+dec_part +' & ')
                else:
                    outfile.write(' & '+sign_str+int_part + ' & ')
            else:
                if ndec[3] != 0:
                    outfile.write('\t'+sign_str+int_part+'.'+dec_part+'\t')
                else:
                    outfile.write('\t'+sign_str+int_part+'\t')
            #Write rho_{xy}
            rho_xy= data[ii][1][0,1]/sigma_x/sigma_y
            if sign(rho_xy) == -1:
                sign_str= '-'
            else:
                sign_str= ''
            int_part=abs(long(rho_xy))
            dec_part= long(round(10**ndec[4]*abs(rho_xy-long(rho_xy))))
            if dec_part >= 10**ndec[4]:
                int_part = int_part+1
                dec_part= dec_part-10**ndec[4]
            int_part= str(int_part)
            if dec_part == 0:
                sign_str=''
            dec_part='%i' % dec_part
            dec_part= dec_part.zfill(ndec[4])
            if latex:
                if ndec[4] != 0:
                    outfile.write(sign_str+int_part+' & '+dec_part)
                else:
                    outfile.write(sign_str+int_part)
            else:
                if ndec[4] != 0:
                    outfile.write(sign_str+int_part+'.'+dec_part)
                else:
                    outfile.write(sign_str+int_part)
            
        if latex:
            outfile.write(r'\\'+'\n')
        else:
            outfile.write('\n')
    #Write the footer
    if latex:
        if allerr:
            outfile.write(r'\tablecomments{The full uncertainty covariance matrix for each data point is given by\\ $\left[\begin{array}{cc} \sigma_x^2 & \rho_{xy}\sigma_x\sigma_y\\\rho_{xy}\sigma_x\sigma_y & \sigma_y^2\end{array}\right]$.}'+'\n')
            outfile.write(r'\label{table:data_allerr}'+'\n')
        else:
            outfile.write(r'\tablecomments{$\sigma_y$ is the uncertainty for the $y$ measurement.}'+'\n')
            outfile.write(r'\label{table:data_yerr}'+'\n')
        outfile.write(r'\enddata'+'\n')
        outfile.write(r'\end{deluxetable}'+'\n')
    outfile.close()
    
    return 0

def read_data(datafilename='data_yerr.dat',allerr=False):
    """read_data_yerr: Read the data from the file into a python structure
    Reads {x_i,y_i,sigma_yi}

    Input:
       datafilename    - the name of the file holding the data
       allerr          - If set to True, read all of the errors

    Output:
       Returns a list {i,datapoint, y_err}, or {i,datapoint,y_err, x_err, corr}

    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    if allerr:
        ncol= 6
    else:
        ncol= 4
    #Open data file
    datafile= open(datafilename,'r')
    #catch-all re that reads numbers
    expr= re.compile(r"-?[0-9]+(\.[0-9]*)?(E\+?-?[0-9]+)?")
    rawdata=[]
    nline= 0
    for line in datafile:
        if line[0] == '#':#Comments
            continue
        nline+= 1
        values= expr.finditer(line)
        nvalue= 0
        for i in values:
            rawdata.append(float(i.group()))
            nvalue+= 1
        if nvalue != ncol:
            print("Warning, number of columns for this record does not match the expected number")
    #Now process the raw data
    out=[]
    for ii in range(nline):
        #First column is the data number
        thissample= []
        thissample.append(rawdata[ii*ncol])
        sample= sc.array([rawdata[ii*ncol+1],rawdata[ii*ncol+2]])
        thissample.append(sample)
        thissample.append(rawdata[ii*ncol+3])
        if allerr:
            thissample.append(rawdata[ii*ncol+4])
            thissample.append(rawdata[ii*ncol+5])
        out.append(thissample)
    return out

# antoher file
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
import scipy.stats as stats
import scipy.linalg as linalg
import math as m

def sample_wishart(n,V,nsamples=1):
    """sample_wishart: Sample a matrix from a Wishart distribution given
    by a shape paramter n and a scale matrix V
    Based on: W. B. Smith and R. R. Hocking, Algorithm AS 53: Wishart
    Variate Generator, Applied Statistic, 21, 341

    W(W|n,V) = |W|^([n-1-p]/2) exp(-Tr[V^(-1)W]/2)/ ( 2^(np/2) |V|^(n/2)
    pi^(p(p-1)/2) Prod_{j=1}^p \Gamma([n+1-j]/2) )
    where p is the dimension of V

    Input:
       n        - shape parameter (> p-1)
       V        - scale matrix
       nsamples - (optional) number of samples desired (if != 1 a list is returned)

    Output:
       a sample of the distribution

    Dependencies:
       scipy
       scipy.stats.chi2
       scipy.stats.norm
       scipy.linalg.cholesky
       math.sqrt

    History:
       2009-05-20 - Written Bovy (NYU)
    """
    #Check that n > p-1
    p= V.shape[0]
    if n < p-1:
        return -1
    #First lower Cholesky of V
    L= linalg.cholesky(V,lower=True)
    if nsamples > 1:
        out= []
    for kk in range(nsamples):
        #Generate the lower triangular A such that a_ii = (\chi2_(n-i+2))^{1/2} and a_{ij} ~ N(0,1) for j < i (i 1-based)
        A= sc.zeros((p,p))
        for ii in range(p):
            A[ii,ii]= m.sqrt(stats.chi2.rvs(n-ii+2))
            for jj in range(ii):
                A[ii,jj]= stats.norm.rvs()
        #Compute the sample X = L A A\T L\T
        thissample= sc.dot(L,A)
        thissample= sc.dot(thissample,thissample.transpose())
        if nsamples == 1:
            return thissample
        else:
            out.append(thissample)
    return out
# antoehr file
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

ex3()