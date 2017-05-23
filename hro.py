# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

def roangles(Imap, Qmap, Umap):
    # Calculates the relative orientation angle between the density structures and the magnetic field.
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Upam - Stokes U map
    # OUTPUTS
    # phi - relative orientation angle between the column density and projected magnetic field.
    
    psi=0.5*np.arctan2(-Umap,Qmap)	
    ex=np.sin(psi)
    ey=np.cos(psi)
    
    grad=np.gradient(Imap, edge_order=2)
    #kernel=Gaussian2DKernel(3)
    #grad=np.gradient(convolve_fft(Imap, kernel))	
    phi=np.arctan2(grad[0]*ex-grad[1]*ey, grad[0]*ey+grad[1]*ex)		
    
    bad=np.logical_or(grad[0]*grad[0]+grad[1]*grad[1]==0., Qmap*Qmap+Umap*Umap==0.).nonzero()	
    phi[bad]=np.sqrt(-1)
    
    return np.abs(phi)

def roparameter(phi, hist, s_phi=15.):
    # Calculate the relative orientation parameter
    # INPUTS
    # phi     - vector with the reference values for the histogram
    # hist    - histogram of relative orientations 
    # s_phi   - range for the definitions of parallel (0 < phi < s_phi and 180-s_phi < phi < 180) or 
    #           perpendicular (90-s_phi < phi < 90+s_phi)
    # OUTPUTS
    # xi 	  - relative orientation parameter
    
    perp=(np.abs(phi-90.)<s_phi).nonzero()
    para=(np.abs(phi-90.)>90.-s_phi).nonzero()
    xi=(np.sum(hist[para])-np.sum(hist[perp]))/float(np.sum(hist[para])+np.sum(hist[perp]))
    
    return xi

def hro(Imap, Qmap, Umap, steps=10, hsize=21, minI=0., outh=[0,4,9]):
    # Calculates the relative orientation angle between the density structures and the magnetic field.
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Upam - Stokes U map
    
    sz=np.shape(Imap)
    phi=roangles(Imap, Qmap, Umap)
    
    hist, bin_edges = np.histogram(Imap[(Imap > minI).nonzero()], bins=100*sz[0])
    bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
    
    chist=np.cumsum(hist)
    pitch=np.max(chist)/float(steps)
    
    hsteps=pitch*np.arange(0,steps+1,1)	
    Isteps=np.zeros(steps+1)
    
    for i in range(0, np.size(Isteps)-1):
        good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
        Isteps[i]=np.min(bin_centre[good])	
    Isteps[np.size(Isteps)-1]=np.max(Imap)
    print(Isteps)
    
    hros=np.zeros([steps,hsize])	
    Smap=0.*Imap
    xi=np.zeros(steps)
    cdens=np.zeros(steps)
    
    for i in range(0, np.size(Isteps)-1):
        good=np.logical_and(Imap>Isteps[i],Imap<Isteps[i+1]).nonzero()
        print(np.size(good))
        hist, bin_edges = np.histogram((180/np.pi)*phi[good], bins=hsize, range=(0.,180.))	
        bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
        hros[i,:]=hist
        Smap[good]=i
        xi[i]=roparameter(bin_centre, hist)
        cdens[i]=np.mean([Isteps[i],Isteps[i+1]])
        #plt.plot(bin_centre, hist)
    
    outsteps=np.size(outh)
    color=iter(cm.cool(np.linspace(0, 1, outsteps)))
    fig=plt.figure()
    for i in range(0, outsteps):
        c=next(color)
        labeltext="%.2f"%Isteps[outh[i]] + r' < $N_{\rm H}/$cm$^{-2}$ < ' + "%.2f"%Isteps[outh[i]+1]
        plt.plot(bin_centre, hros[outh[i],:], '-', linewidth=2, c=c, label=labeltext) #drawstyle
    plt.xlabel(r'cos($\phi$)')
    plt.legend()
    plt.show()	
    
    fig=plt.figure()
    plt.plot(cdens, xi, '-', linewidth=2, c=c)
    plt.axhline(y=0., c='k', ls='--')
    plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
    plt.ylabel(r'$\zeta$')
    #plt.savefig(prefix + '-' + 'ROvsLogNH' + 'Thres' + "%d" % (thr) + '.png')
    plt.show()
    
    import pdb; pdb.set_trace()
    
    return hros, Isteps


def main(args=None):

    if res.prnt:
    print('Primes: {0}'.format(primes))

Imap=fits.open('data/Taurusfwhm5_logNHmap.fits')
Qmap=fits.open('data/Taurusfwhm10_Qmap.fits')
Umap=fits.open('data/Taurusfwhm10_Umap.fits')
hro(Imap[0].data, Qmap[0].data, Umap[0].data, minI=21.0)

