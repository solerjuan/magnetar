# This file is part of AstroHOG
#
# Copyright (C) 2019-2025 Juan Diego Soler

import sys
import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#from nose.tools import assert_equal, assert_true
      
#import sklearn
import healpy as hp
from tqdm import tqdm
   
#from statests import *
   
sigma2fwhm=2.*np.sqrt(2.*np.log(2.))
fwhm2sigma=1/sigma2fwhm

# -------------------------------------------------------------------------------------
def gaussian(x, mu, sig, limg=1e-3):
   
    gfunc=1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    gfunc[(gfunc < np.max(gfunc)*limg).nonzero()]=0.

    return gfunc

# -------------------------------------------------------------------------------------
def gradienthp(hpmap, niter=3, ksz=None, nsideout=None):

   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
    hpmap   : healpix map 
    niter   : 
    ksz     : Size of the derivative kernel in degrees
    nsideout: nside of the output map
 
   Returns
   -------
    circstats:  Statistics describing the correlation between the input images.

   """

   if (ksz is None):
      ksz=2*np.rad2deg(hp.nside2resol(hp.npix2nside(np.size(hpmap))))

   lmax=int(np.ceil(180/ksz))

   if (nsideout is None):
      nsideout=hp.npix2nside(np.size(hpmap))

   inhpmap=hpmap.copy()-np.nanmean(hpmap)
   
   #alm1=hp.sphtfunc.anafast(inmap1, iter=niter, alm=True, lmax=lmax, pol=False, use_weights=False, gal_cut=gal_cut, use_pixel_weights=False)
   #smap1, dmap1dtheta, dmap1dphi = hp.sphtfunc.alm2map_der1(alm1[1], hp.npix2nside(np.size(map1)), lmax=lmax, mmax=None)

   #alm=hp.sphtfunc.map2alm(inhpmap, iter=niter) 
   alm=hp.sphtfunc.map2alm(inhpmap, iter=niter, use_pixel_weights=True)
   clm=hp.sphtfunc.alm2cl(alm)
   ell=np.arange(np.size(clm))+1

   g1=gaussian(np.arange(np.size(clm)), 0., lmax)
   clip=g1/np.max(g1)
   #clip=np.ones(lmax+1)
   alm_clipped=hp.almxfl(alm, clip)
   clm_clipped=hp.sphtfunc.alm2cl(alm_clipped)

   smap, dmapdtheta, dmapdphi = hp.sphtfunc.alm2map_der1(alm_clipped, nsideout)
   gradmap=np.sqrt(dmapdtheta**2+dmapdphi**2)

   output={'dtheta': dmapdtheta, 'dphi': dmapdphi, 'smap': smap, 'gradmap': gradmap}

   return output

# -------------------------------------------------------------------------------------
def gradPsi(Qmap, Umap, ksz=1.0, niter=3):

   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
    Imap    : healpix map 
    Qmap    :
    Umap    : 
    ksz     : Derivative kernel width in degrees
 
   Returns
   -------
    circstats:  Statistics describing the correlation between the input images.

   """ 

   Pmap=np.sqrt(Qmap**2+Umap**2)
   # Calculating GradPsi ------------------------------------------
   output=gradienthp(Qmap/Pmap, niter=niter, ksz=ksz)
   dQoverPdtheta=output['dtheta']
   dQoverPdphi=output['dphi']

   output=gradienthp(Umap/Pmap, niter=niter, ksz=ksz)
   dUoverPdtheta=output['dtheta']
   dUoverPdphi=output['dphi']

   dpsidtheta=np.sqrt(dQoverPdtheta**2+dUoverPdtheta**2)
   dpsidphi=np.sqrt(dQoverPdphi**2+dUoverPdphi**2)
   gradpsi=np.sqrt(dQoverPdtheta**2+dUoverPdtheta**2+dQoverPdphi**2+dUoverPdphi**2)

   return gradpsi 


