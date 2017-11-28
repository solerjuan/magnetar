#
# This file is part of Magnetar
#
# Copyright (C) 2013-2017 Juan Diego Soler

import sys
import numpy as np
from astropy.io import fits
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

#sys.path.append('/Users/soler/Documents/magnetar/')
from hro import *
from bvisual import *

from astropy.wcs import WCS
from reproject import reproject_interp

import os
import imageio

indir='/Users/soler/Documents/magnetar/data/'
prefix='Taurusfwhm10'

# ================================================================================================================================
def exampleHRO2D():

   hdu=fits.open(indir+'Taurusfwhm5_logNHmap.fits')
   logNHmap=hdu[0].data
   hdu=fits.open(indir+prefix+'_Qmap.fits')
   Qmap=hdu[0].data
   hdu=fits.open(indir+prefix+'_Umap.fits')
   Umap=hdu[0].data

   hro(logNHmap, Qmap, Umap, minI=20.7)

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
   ax1.imshow(licmap, origin='lower', alpha=0.4, cmap='binary', clim=[1.25*np.min(licmap),0.75*np.max(licmap)])
   arrows=plt.quiver(x, y, ux, uy, units='width', color='black', pivot='tail', headlength=0, headwidth=0) 
   
   plt.colorbar(im)
   plt.show()

# ================================================================================================================================
exampleHRO2D();
exampleVisualization();


