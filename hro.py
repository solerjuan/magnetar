# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel


# ===================================================================================================
def roangles(Imap, Qmap, Umap, ksz=1):
    # Calculates the relative orientation angle between the density structures and the magnetic field following the method
    # presented in Soler, et al. ApJ 774 (2013) 128S
    #
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Upam - Stokes U map
    # OUTPUTS
    # phi - relative orientation angle between the column density and projected magnetic field.
    
    psi=0.5*np.arctan2(-Umap,Qmap)	
    ex=np.sin(psi)
    ey=np.cos(psi)
    
    if (ksz > 1):
       kernel=Gaussian2DKernel(ksz)
       grad=np.gradient(convolve_fft(Imap, kernel))
    else:
       grad=np.gradient(Imap, edge_order=2)

    phi=np.arctan2(grad[0]*ex-grad[1]*ey, grad[0]*ey+grad[1]*ex)		
    bad=np.logical_or(grad[0]*grad[0]+grad[1]*grad[1]==0., Qmap*Qmap+Umap*Umap==0.).nonzero()	
    phi[bad]=np.nan
    
    return np.arctan(np.tan(phi))


# ===================================================================================================
def roparameter(phi, hist, s_phi=20.):
    # Calculate the relative orientation parameter $\xi$ as defined in Planck intermediate results. XXXV. A&A 586A (2016) 138P.
    #
    # INPUTS
    # phi     - vector with the reference values for the histogram
    # hist    - histogram of relative orientations 
    # s_phi   - range for the definitions of parallel (0 < phi < s_phi and 180-s_phi < phi < 180) or 
    #           perpendicular (90-s_phi < phi < 90+s_phi)
    # OUTPUTS
    # xi 	  - relative orientation parameter

    perp=(np.abs(phi) > 90.-s_phi).nonzero()
    para=(np.abs(phi) < s_phi).nonzero()
    xi=float(np.mean(hist[para])-np.mean(hist[perp]))/float(np.mean(hist[para])+np.mean(hist[perp]))
    
    return xi


# ===================================================================================================
def projRS(phi):
    # Calculate the projected Rayleight statistics as defined in Jow, et al. MNRAS (2018) in press.
    #
    # INPUTS
    # phi      - relative orientation angles defined between -pi/2 and pi/2
    #
    # OUTPUTS
    # psr      - projected Rayleigh statistic
    # s_prs    - 

    angles=2.*phi

    Zx=np.sum(np.cos(angles))/np.sqrt(np.size(angles)/2.)
    temp=np.sum(np.cos(angles)*np.cos(angles))
    s_Zx=np.sqrt((2.*temp-Zx*Zx)/np.size(angles))

    Zy=np.sum(np.sin(angles))/np.sqrt(np.size(angles)/2.)
    temp=np.sum(np.sin(angles)*np.sin(angles))
    s_Zx=np.sqrt((2.*temp-Zy*Zy)/np.size(angles))

    meanPhi=0.5*np.arctan2(Zy, Zx)

    return Zx, s_Zx


# ===================================================================================================
def hro(Imap, Qmap, Umap, steps=10, hsize=15, minI=0., outh=[0,4,9]):
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
    
    hros=np.zeros([steps,hsize])	
    Smap=0.*Imap
    xi=np.zeros(steps)
    prs=np.zeros(steps)
    s_prs=np.zeros(steps)
    cdens=np.zeros(steps)
    
    for i in range(0, np.size(Isteps)-1):
        good=np.logical_and(Imap>Isteps[i],Imap<Isteps[i+1]).nonzero()
 
        hist, bin_edges = np.histogram((180/np.pi)*phi[good], bins=hsize, range=(-90.,90.))	
        bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

        hros[i,:]=hist
        cdens[i]=np.mean([Isteps[i],Isteps[i+1]])
        Smap[good]=i

        xi[i]=roparameter(bin_centre, hist)

        TEMPprs, TEMPs_prs = projRS(phi[good])      
        prs[i]=TEMPprs
        s_prs[i]=TEMPs_prs

        print(i)

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

    # --------------------------------------------------------------    
    fig=plt.figure()
    plt.plot(cdens, xi, '-', linewidth=2, c='red')
    plt.axhline(y=0., c='k', ls='--')
    plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
    plt.ylabel(r'$\zeta$')
    plt.show()

    # --------------------------------------------------------------
    fig=plt.figure()
    plt.plot(cdens, prs, '-', linewidth=2, c='blue')
    plt.errorbar(cdens, prs, yerr=s_prs, c='blue', fmt='o')
    plt.axhline(y=0., c='k', ls='--')
    plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
    plt.ylabel(r'$Z_{x}$')
    plt.show()
    
    return Isteps, xi


