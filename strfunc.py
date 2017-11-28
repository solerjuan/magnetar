# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

# ================================================================================================================================
def strfunclist(xpos, ypos, psi, nsteps=10, EqualNbins=False):

    #Fix to include uncertainties
    s_psi=5.*np.pi/180.

    npos=np.size(xpos)
    
    distances=np.zeros([npos,npos])
    deltapsi=np.zeros([npos,npos])
    deltapsi2=np.zeros([npos,npos])  
    cospsi=np.zeros([npos,npos])
    sinpsi=np.zeros([npos,npos])

    qint=np.sin(psi)**2 - np.cos(psi)**2
    uint=-2.*np.cos(psi)*np.sin(psi)

    for i in range(0, npos):
       
        deltax=xpos[i]-xpos
        deltay=ypos[i]-ypos
        dist=np.sqrt(deltax**2 + deltay**2)
        distances[:,i]=dist
        distances[i:npos,i]=0.   
   
        deltapsi[i,:]=0.5*np.arctan2(qint[i]*uint-qint*uint[i], qint[i]*qint+uint[i]*uint)
        deltapsi2[i,:]=(deltapsi[i,:])**2
        cospsi[i,:]=np.cos(deltapsi[i,:])
        sinpsi[i,:]=np.sin(deltapsi[i,:])

    plt.imshow(distances, origin='lower', interpolation='none')
    plt.show()

    plt.imshow((180./np.pi)*np.sqrt(deltapsi2), origin='lower', interpolation='none')
    plt.colorbar()
    plt.show()
    #import pdb; pdb.set_trace()

    if (EqualNbins):
        # Invert the distribution of distances     
        hist, bin_edges = np.histogram(distances[(distances > 0.).nonzero()], bins=10000)
        bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
        chist=np.cumsum(hist)
        pitch=np.max(chist)/float(nsteps)

        hsteps=pitch*np.arange(0,nsteps+1,1)
        dsteps=np.zeros(nsteps+1)
     
        for i in range(0, np.size(dsteps)-1):
            good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
            dsteps[i]=np.min(bin_centre[good])
        dsteps[np.size(dsteps)-1]=np.max(distances)

    else:
        dpitch=(np.max(distances[(distances > 0.).nonzero()])-np.min(distances[(distances > 0.).nonzero()]))/float(nsteps)
        dsteps=np.arange(np.min(distances[(distances > 0.).nonzero()]), nsteps, dpitch)

    print(dsteps)
    #import pdb; pdb.set_trace()

    # -------------------------------------------------------------------------------------------------------------
    npairs=np.zeros(nsteps)
    lag=np.zeros(nsteps)

    structFunc=np.zeros(nsteps)
    s_structFunc=np.zeros(nsteps)

    cosFunc=np.zeros(nsteps) 
    s_cosFunc=np.zeros(nsteps)
 
    for i in range(0, nsteps):
        good=np.logical_and(distances>dsteps[i],distances<dsteps[i+1]).nonzero()
        print(np.size(good), ' between ', dsteps[i], 'and', dsteps[i+1])
        lag[i]=0.5*(dsteps[i+1]+dsteps[i])
        npairs[i]=float(np.size(good))

        if(npairs[i] > 0.):
            tempdeltapsi2=deltapsi2[good]
            #tempdeltapsi2=np.arctan(np.tan(np.sqrt(deltapsi2[good])))**2
            #print(np.max((180/np.pi)*tempdeltapsi2))
            structFunc[i]=np.sqrt(np.mean(tempdeltapsi2))
                
            A1=(np.sum(deltapsi)**2)*(s_psi**2)
            A2=np.sum(deltapsi2*s_psi**2)
            s_structFunc[i]=np.sqrt(A1+A2)/(npairs[i]*structFunc[i])

            cosFunc[i]=np.sum(cospsi[good])/np.size(good)

    return lag, structFunc, cosFunc


# =================================================================================================================================
def strfunc(Imap, Qmap, Umap, pxsz=1., nsteps=10, beamfwhm=1.):
    # Calculates the relative orientation angle between the density structures and the magnetic field.
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Umap - Stokes U map
    #

# Convert polarization maps into lists
    sz=np.shape(Imap)
    psi=0.5*np.arctan2(Umap,Qmap)
    x=pxsz*np.arange(sz[0])/float(sz[0])
    y=pxsz*np.arange(sz[1])/float(sz[1])

# Computing coordinate grid
    xv, yv = np.meshgrid(x, y) #, sparse=False, indexing='ij')

# Compute structure function using the lists
    lag, structFunc, cosFunc = strfunclist(xv.ravel(), yv.ravel(), psi.ravel(), nsteps=nsteps, EqualNbins=True)

    plt.figure()
    plt.plot(lag, (180/np.pi)*structFunc, 'ro')
    plt.show()

    good4fit=(lag > 0.3).nonzero()  
    import pdb; pdb.set_trace() 
    #z = np.polyfit(lag[good4fit]**2, 1.-cosFunc[good4fit], 2)

    plt.figure()
    plt.plot(lag, 1.-cosFunc, 'bo')
    plt.plot(lag, z, 'k')
    plt.show()

    import pdb; pdb.set_trace()

from astropy.convolution import convolve_fft

#npix=60
#Imap=np.random.rand(npix,npix)
#Qmap=np.random.rand(npix,npix)-0.5
#Umap=np.random.rand(npix,npix)-0.5

Imap=fits.open('data/Taurusfwhm5_logNHmap.fits')[0].data
Qmap=fits.open('data/Taurusfwhm10_Qmap.fits')[0].data
Umap=fits.open('data/Taurusfwhm10_Umap.fits')[0].data

#import pdb; pdb.set_trace()
pxksz=5
strfunc(Imap, Qmap, Umap, nsteps=20, pxsz=1.)
#strfunc(Imap, convolve_fft(Qmap, Gaussian2DKernel(pxksz), boundary='wrap'), convolve_fft(Umap, Gaussian2DKernel(pxksz), boundary='wrap'), nsteps=npix, pxsz=1.)
#strfunc(Imap, convolve_fft(Qmap, Gaussian2DKernel(pxksz), boundary='fill'), convolve_fft(Umap, Gaussian2DKernel(pxksz), boundary='fill'), nsteps=npix, pxsz=1.)

