# Example of the histogram of orientated gradients (HRO) method
# applied to the column density and magnetic field orientation observations toward the Taurus molecular cloud
# See Planck Collaboration XXXV. A&A, 586 (2016) A138
#
# This file is part of Magnetar
#
# Copyright (C) 2013-2023 Juan Diego Soler

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')
from hro import *
from bvisual import *

from astropy.wcs import WCS
from reproject import reproject_interp

import os
import imageio

indir='../data/'
prefix='Taurusfwhm10'

# Loading Taurus Stokes I, Q, and U maps from Planck at 353 GHz

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

# Planck beam size
fwhm=10.0 # arcmin

# Setting the size of the derivative kernel

sz=np.shape(logNHmap)
deltab=60.*hdrREF['CDELT2'] # in arcmin 
ksz=10. # arcmin
pxksz=ksz/deltab

# Setting the number column density bins
steps=25 # number of histograms

# Setting the statistical weights to account for the beam sampling
weights=np.ones(sz)*(deltab/ksz)**2

# Simple mask definition
mask=np.ones_like(logNHmap)
mask[(logNHmap < 21.2)]=0.

# HRO calculation
outputhro = hroLITE(logNHmap, Qmap, -Umap, steps=20, minI=np.nanmin(logNHmap), w=weights, mask=mask, ksz=pxksz)

# Map of relative orientation angles 
fig = plt.figure(figsize=(10.0,10.0))
plt.rc('font', size=12)
ax1=plt.subplot(111, projection=WCS(hdrREF))
im=ax1.imshow(np.abs(np.rad2deg(outputhro['Amap'])), origin='lower', interpolation='none', cmap='jet')
ax1.contour(logNHmap, origin='lower', levels=[np.mean(logNHmap)+1.0*np.std(logNHmap),np.mean(logNHmap)+2.0*np.std(logNHmap)], colors='black', linewidths=2.0)
ax1.coords[0].set_axislabel(r'$l$')
ax1.coords[1].set_axislabel(r'$b$')
cbar=fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.set_title(r'$\phi$ [deg]')
plt.show()

# Setting the center of the column density bins
isteps=outputhro['csteps']
icentres=0.5*(isteps[0:np.size(isteps)-1]+isteps[1:np.size(isteps)])

# Reproducing top-left panel of Figure 7 in Planck Collaboration XXXV. A&A, 586 (2016) A138

fig = plt.figure(figsize=(10.0,5.0))
plt.rc('font', size=12)
ax1=plt.subplot(111)
ax1.plot(icentres, outputhro['xi'], color='orange')
ax1.axhline(y=0, color='grey', alpha=0.5, linestyle='dashed')
ax1.tick_params(axis='y', labelrotation=90)
ax1.set_xlabel(r'log($N_{\rm H}/$cm$^{2}$)')
ax1.set_ylabel(r'$\xi$')
plt.show()





