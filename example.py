#
# This file is part of Magnetar
#
# Copyright (C) 2013-2018 Juan Diego Soler

import sys
import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

from hro import *
from bvisual import *

from astropy.wcs import WCS
from reproject import reproject_interp

import os
import imageio

indir='data/'
prefix='Taurusfwhm10'

# ================================================================================================================================
def exampleHRO2D():

   hdu=fits.open(indir+'Taurusfwhm5_logNHmap.fits')
   logNHmap=hdu[0].data
   hdrREF=hdu[0].header
   hdu.close()
   hdu=fits.open(indir+prefix+'_Qmap.fits')
   Qmap=hdu[0].data
   hdu.close()
   hdu=fits.open(indir+prefix+'_Umap.fits')
   Umap=hdu[0].data
   hdu.close()
 
   sz=np.shape(logNHmap) 
   deltab=60.*hdrREF['CDELT2'] # in arcmin 
   ksz=15. # in arcmin
   steps=25 # number of histograms
   weights=np.ones(sz)*(deltab/ksz)**2

   isteps, xi, prs, meanphi = hro(logNHmap, Qmap, Umap, steps=20, minI=np.min(logNHmap), w=weights)
  
   plt.plot(isteps, xi, color='orange')
   plt.show()
 
   plt.plot(isteps, prs, color='orange')
   plt.show()

   plt.plot(isteps, np.abs(meanphi)*180./np.pi, color='orange')
   plt.show()

# ================================================================================================================================
def exampleVisualization():

   hdu=fits.open(indir+'Taurusfwhm5_logNHmap.fits')
   Imap=hdu[0].data
   refhdr1=hdu[0].header 
   hdu=fits.open(indir+prefix+'_Qmap.fits')
   Qmap=hdu[0].data
   hdu=fits.open(indir+prefix+'_Umap.fits')
   Umap=hdu[0].data

   psi=np.arctan2(-Umap,Qmap)

   ex=np.cos(psi)
   ey=np.sin(psi)
   bx=ey
   by=-ex

   sz=np.shape(Qmap)
   length=int(0.1*sz[0]) 

   licmap=lic(bx, by, length=length, niter=1)
   x, y, ux, uy = vectors(Imap, bx, by, pitch=15)

   ax1=plt.subplot(1,1,1, projection=WCS(refhdr1))
   im=ax1.imshow(Imap, origin='lower', cmap=planckct())
   ax1.imshow(licmap, origin='lower', alpha=0.4, cmap='binary', clim=[np.mean(licmap)-np.std(licmap),np.mean(licmap)+np.std(licmap)])
   arrows=plt.quiver(x, y, ux, uy, units='width', color='black', pivot='middle', headlength=0, headwidth=0)  
   plt.colorbar(im)
   plt.show()
   import pdb; pdb.set_trace()

# ================================================================================================================================
#exampleVisualization();
exampleHRO2D();


