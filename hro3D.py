# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2020 Juan Diego Soler

import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

from scipy import ndimage


def roangles3D(dens, Bx, By, Bz, mode='nearest', pxksz=3):
    """
    Calculates the relative orientation angles between the density structures 
        and the magnetic field.
    
    
        Parameters
        ----------
        dens : numpy.ndarray. 
             density field.
        Bx, y, z : float or numpy.ndarray
            	 magnetic field strength in each direction.
    
        Returns
        -------
        numpy.ndarray of the relative angles between the density structure and 
    	    and magnetic field structure.
    
        Notes
        -----
    	...
    
        References
        ----------
        .. [1] Soler et al 2013 ...
    
        Examples
        --------
    	...
    """
    # Calculates the relative orientation angles between the density structures and the magnetic field.
    # INPUTS
    # dens - regular cube with the values of density 
    # Bx   -
    # By   -
    # Bz   -
    #
    # OUTPUTS
    #
    #
    
    #gx=grad[1]; gy=grad[0]; gz=grad[2];
    gx=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,0,1], mode=mode)
    gy=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[0,1,0], mode=mode)
    gz=ndimage.filters.gaussian_filter(dens, [pxksz, pxksz, pxksz], order=[1,0,0], mode=mode)   

    normgrad=np.sqrt(gx*gx+gy*gy+gz*gz)
    normb   =np.sqrt(Bx*Bx+By*By+Bz*Bz)
    
    zerograd=(normgrad==0.).nonzero()	
    zerob   =(normb   ==0.).nonzero()
    
    cross=np.sqrt((gy*Bz-gz*By)**2+(gx*Bz-gz*Bx)**2+(gx*By-gy*Bx)**2)
    dot  =gx*Bx+gy*By+gz*Bz	
    
    # The cosine of the angle between the iso-density and B is the sine of the angle between
    # the density gradient and B.	
    cosphi=dot/(normgrad*normb)   
 
    cosphi[(normgrad == 0.).nonzero()]=np.nan
    cosphi[(normb    == 0.).nonzero()]=np.nan
    
    return cosphi


def equibins(dens, steps=10, mind=0.):
    """
    ...
    
    
        Parameters
        ----------
        dens : numpy.ndarray. 
             density field.
        steps: int
             ...
        nubd : float
             ...
    
        Returns
        -------
        ...
    
        Notes
        -----
    	...
    
        References
        ----------
        .. [1] Soler et al 2013 ...
    
        Examples
        --------
    	...
    """
    # Calculates the relative orientation angles between the density structures and the magnetic field.
    # INPUTS
        # dens - regular cube with the values of density 
    # steps -
    # mind - 
    
    sz=np.shape(dens)
    hist, bin_edges = np.histogram(dens[(dens > mind).nonzero()], bins=10*sz[0]*sz[1])
    bin_centre     =0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
    
    chist=np.cumsum(hist)
    pitch=np.max(chist)/float(steps)
    
    hsteps=pitch*np.arange(0,steps+1,1)
    dsteps=np.zeros(steps+1)
    
    for i in range(0, np.size(dsteps)-1):	                
        good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
        dsteps[i]=np.min(bin_centre[good])
    
    dsteps[np.size(dsteps)-1]=np.max(dens)
    
    return dsteps

def roparameter(cosphi, hist, s_cosphi=0.25):
    """
    ...
    
        Parameters
        ----------
        cosphi : ...
             
        hist   : ...
             
       
        s_cosphi : ...
             
        
        Returns
        -------
        ...
    
    """
    	
    perp=(np.abs(cosphi)>1.-s_cosphi).nonzero()
    para=(np.abs(cosphi)<s_cosphi).nonzero()
    
    xi=(np.sum(hist[para])-np.sum(hist[perp]))/float(np.sum(hist[para])+np.sum(hist[perp]))
    
    return xi

def hro3D(dens, Bx, By, Bz, steps=10, hsize=21, mind=0, outh=[0,4,9], pxksz=3, label=r'$n$'):
    """
    Calculate the histogram of relative orientations (HRO) in three-dimensional data.

        Parameters
        ----------
        dens   : ...

        Bx,y,z : ...

        steps  : ...

        hsize  : ...

        mind   : ...

        outh   : ...

        Returns
        -------
        hro   : ...

        cdens : ...

        zeta  : ...

    """

    cosphi=roangles3D(dens, Bx, By, Bz, pxksz=pxksz)
    dsteps=equibins(dens, steps=steps, mind=mind)

    hros   =np.zeros([steps,hsize])
    cdens  =np.zeros(steps)
    xi     =np.zeros(steps)
    meancos=np.zeros(steps)
    scube = 0.*dens

    for i in range(0, np.size(dsteps)-1):
        good=np.logical_and(dens>dsteps[i],dens<dsteps[i+1]).nonzero()
        hist, bin_edges=np.histogram(cosphi[good], bins=hsize, range=(-1.,1.))	
        bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
        hros[i,:]=hist
        scube[good]=i
        cdens[i]=np.mean([dsteps[i],dsteps[i+1]])
        xi[i]=roparameter(bin_centre, hist)
        meancos[i]=np.mean(cosphi[good]) 

    outsteps = np.size(outh)
    color    = iter(cm.cool(np.linspace(0, 1, outsteps)))

    fig = plt.figure(figsize=(6.0,6.0))
    plt.rc('font', size=10)
    ax1=plt.subplot(111)
    for i in range(0, outsteps):
        c = next(color)
        labeltext = str(np.round(dsteps[outh[i]],2))+' < '+label+' < '+str(np.round(dsteps[outh[i]+1],2)) 
        ax1.plot(bin_centre, hros[outh[i],:], '-', linewidth=2, c=c, label=labeltext) #drawstyle
    ax1.set_xlabel(r'cos($\phi$)')
    ax1.set_ylabel('Counts')
    ax1.tick_params(axis='y', labelrotation=90) 
    ax1.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8.0,4.0))
    plt.rc('font', size=10)
    ax1=plt.subplot(111)
    ax1.semilogx(cdens, xi, 'o-', linewidth=2, color='blue')
    ax1.tick_params(axis='y', labelrotation=90) 
    ax1.axhline(y=0., c='k', ls='--')	
    ax1.set_xlabel(r'log$_{10}$ ($n_{\rm H}/$cm$^{-3}$)')
    ax1.set_ylabel(r'$\xi$')
    plt.tight_layout()
    #plt.savefig(prefix + '-' + 'ROvsLogNH' + 'Thres' + "%d" % (thr) + '.png')
    plt.show()

    fig = plt.figure(figsize=(8.0,4.0))
    plt.rc('font', size=10)
    ax1=plt.subplot(111)
    ax1.semilogx(cdens, meancos, 'o-', linewidth=2, color='blue')
    ax1.tick_params(axis='y', labelrotation=90)
    ax1.axhline(y=0., c='k', ls='--')  
    ax1.set_xlabel(r'log$_{10}$ ($n_{\rm H}/$cm$^{-3}$)')
    ax1.set_ylabel(r'$\left<\cos\theta\right>$')
    plt.tight_layout()
    plt.show()
  
    return hros, cdens, xi


# Testing the hro3D. should go in the test script or something.
#def main(args=None):
#
#    	if res.prnt:
#		print('Primes: {0}'.format(primes))
#
#from astropy.convolution import Gaussian2DKernel
#g2D=Gaussian2DKernel(10)
#sz=np.shape(g2D)
#dens=np.dstack([g2D]*sz[0])
#
#grad=np.gradient(dens, edge_order=2)
##Bx=grad[1]
##By=grad[0]
##Bz=grad[2]
#Bx=np.random.uniform(low=-1., high=1., size=np.shape(dens))
#By=np.random.uniform(low=-1., high=1., size=np.shape(dens))
##Bz=np.random.uniform(low=-1., high=1., size=np.shape(dens))
##Bx=0.*dens
##By=0.*dens
#Bz=0.*dens
#hros, cdens, zeta = hro3D(dens, Bx, By, Bz, mind=np.mean(dens))
