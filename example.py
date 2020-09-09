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
   ksz=15. # arcmin
   pxksz=ksz/deltab
   steps=25 # number of histograms
   weights=np.ones(sz)*(deltab/ksz)**2

   # mask definition
   mask=np.ones_like(logNHmap)
   mask[(logNHmap < np.mean(logNHmap))]=0.

   # HRO calculation
   outputhro = hroLITE(logNHmap, Qmap, Umap, steps=20, minI=np.min(logNHmap), w=weights, mask=mask, ksz=pxksz)

   # --------------------------------
   # Calculation of the polarization angle 
   psi=0.5*np.arctan2(-Umap,Qmap) 
   ex=-np.sin(psi); ey=np.cos(psi)
   # 90 degrees rotation to obtain the magnetic field orientation
   bx=ey; by=-ex                

   sImap=ndimage.filters.gaussian_filter(logNHmap, [ksz, ksz], order=[0,0], mode='nearest')
   dIdx =ndimage.filters.gaussian_filter(logNHmap, [ksz, ksz], order=[0,1], mode='nearest')
   dIdy =ndimage.filters.gaussian_filter(logNHmap, [ksz, ksz], order=[1,0], mode='nearest')
   
   # ------------------------------------------------------------------------------------
   vecpitch=20
   xx, yy, gx, gy = vectors(logNHmap, dIdx, dIdy, pitch=vecpitch)
   xcoord=hdrREF['CRVAL1']+(xx-hdrREF['CRPIX1'])*hdrREF['CDELT1']
   ycoord=hdrREF['CRVAL2']+(yy-hdrREF['CRPIX2'])*hdrREF['CDELT2']

   xx, yy, ux, uy = vectors(logNHmap, bx, by, pitch=vecpitch)
   xcoord=hdrREF['CRVAL1']+(xx-hdrREF['CRPIX1'])*hdrREF['CDELT1']
   ycoord=hdrREF['CRVAL2']+(yy-hdrREF['CRPIX2'])*hdrREF['CDELT2']

   fig = plt.figure(figsize=(8.0,7.0))
   plt.rc('font', size=10)
   ax1=plt.subplot(111, projection=WCS(hdrREF))
   #im=ax1.imshow(logNHmap, origin='lower', cmap='viridis')
   im=ax1.imshow(np.abs(np.rad2deg(outputhro['Amap'])), origin='lower', cmap='viridis')
   ax1.contour(sImap, origin='lower', levels=[0.010,0.02,0.03,0.06,0.125,0.25], colors='black', linewidths=0.5)
   ax1.quiver(xx, yy, gx, gy, units='width', color='red', pivot='middle', scale=25., headlength=0, headwidth=1, alpha=0.8, transform=ax1.get_transform('pixel'))
   ax1.quiver(xx, yy, ux, uy, units='width', color='black', pivot='middle', scale=25., headlength=0, headwidth=1, transform=ax1.get_transform('pixel'))
   ax1.coords[0].set_axislabel(r'$l$')
   ax1.coords[1].set_axislabel(r'$b$')
   cbar=fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
   cbar.ax.set_title(r'$\phi$')
   plt.show() 
   import pdb; pdb.set_trace() 

   isteps=outputhro['csteps']
   icentres=0.5*(isteps[0:np.size(isteps)-1]+isteps[1:np.size(isteps)])

   fig = plt.figure(figsize=(8.0,4.5))
   plt.rc('font', size=10)
   ax1=plt.subplot(111)
   ax1.plot(icentres, outputhro['xi'], color='orange')
   ax1.axhline(y=0, color='grey', alpha=0.5)
   ax1.set_xlabel(r'log($N_{\rm H}/$cm$^{2}$)')
   ax1.set_ylabel(r'$\xi$')
   plt.show()
 
   #plt.plot(isteps, prs, color='orange')
   #plt.show()

   #plt.plot(isteps, np.abs(meanphi)*180./np.pi, color='orange')
   #plt.show()

# ================================================================================================================================
def exampleVisualization():

   hdu=fits.open(indir+'Taurusfwhm5_logNHmap.fits')
   Imap=hdu[0].data
   refhdr1=hdu[0].header 
   hdu=fits.open(indir+prefix+'_Qmap.fits')
   Qmap=hdu[0].data
   hdu=fits.open(indir+prefix+'_Umap.fits')
   Umap=hdu[0].data

   psi=0.5*np.arctan2(-Umap,Qmap)

   ex=np.cos(psi)
   ey=np.sin(psi)
   bx=ey
   by=-ex

   sz=np.shape(Qmap)
   length=int(0.05*sz[0]) 

   licmap=lic(bx, by, length=length, niter=3)
   x, y, ux, uy = vectors(Imap, bx, by, pitch=15)

   ax1=plt.subplot(1,1,1, projection=WCS(refhdr1))
   im=ax1.imshow(Imap, origin='lower', cmap=planckct())
   ax1.imshow(licmap, origin='lower', alpha=0.4, cmap='binary', clim=[np.mean(licmap)-np.std(licmap),np.mean(licmap)+np.std(licmap)])
   arrows=plt.quiver(x, y, ux, uy, units='width', color='black', pivot='middle', headlength=0, headwidth=0)  
   plt.colorbar(im)
   plt.show()
   import pdb; pdb.set_trace()

# ==============================================================================================================
def exampleStructureFunction():

   hdu=fits.open('../PlanckHerschelGouldBelt/Orion/Orionfwhm10_Imap.fits')
   Imap0=hdu[0].data
   hdrI=hdu[0].header
   hdu=fits.open('../PlanckHerschelGouldBelt/Orion/Orionfwhm10_Qmap.fits')
   Qmap0=hdu[0].data
   hdu=fits.open('../PlanckHerschelGouldBelt/Orion/Orionfwhm10_Umap.fits')
   Umap0=hdu[0].data

   y0=50
   y1=250
   x0=210
   x1=410

   Imap=Imap0[y0:y1,x0:x1]
   Qmap=Qmap0[y0:y1,x0:x1]
   Umap=Umap0[y0:y1,x0:x1]
   mask=1+0*Qmap
   #mask[(Imap < 21.5).nonzero()]=0

   fig = plt.figure(figsize=(4.0, 4.0), dpi=150)
   plt.rc('font', size=8)
   ax1=plt.subplot(111, projection=WCS(hdrI))
   ax1.set_xlim(x0,x1)
   ax1.set_ylim(y0,y1)
   ax1.imshow(np.log10(Imap0), origin='lower', cmap=planckct())
   ax1.coords[0].set_axislabel('Galactic Longitude')
   ax1.coords[1].set_axislabel('Galactic Latitude')
   plt.savefig('/Users/soler/Downloads/OrionA_Imap.png', bbox_inches='tight')
   plt.close()
 
   psi=0.5*np.arctan2(-Umap,Qmap)
   sigmapsi=circvar(psi)
   print('SigmaPsi',sigmapsi*180.0/np.pi)

   #lags=np.arange(2.,40.,4.)
   lags=np.arange(2.,50.,2.)
   s2=np.zeros(np.size(lags))

   for i in range(0, np.size(lags)):
      s2[i]=structurefunction(Qmap, Umap, mask=mask, lag=lags[i], s_lag=1.0)
      print(60.*np.abs(hdrI['CDELT1'])*lags[i],s2[i]*180.0/np.pi)

   physlags=60.*np.abs(hdrI['CDELT1'])*lags
   fitrange=(physlags > 10.*2.).nonzero()
   z = np.polyfit(physlags[fitrange]**2, s2[fitrange]**2, 2)
   poly = np.poly1d(z)
   print(np.sqrt(poly(0.0))*180.0/np.pi)

   fig = plt.figure(figsize=(6.0, 4.0), dpi=150)
   plt.rc('font', size=8)
   plt.plot(60.*np.abs(hdrI['CDELT1'])*lags, s2*180.0/np.pi, color='black')
   plt.plot(60.*np.abs(hdrI['CDELT1'])*lags, np.sqrt(poly(physlags**2))*180.0/np.pi, color='orange')
   plt.axhline(y=sigmapsi*180.0/np.pi, color='cyan')
   plt.xlabel(r'$\delta$ [arcmin]')
   plt.ylabel(r'$S_{2}(\delta)$ [deg]')
   #plt.show()
   plt.savefig('/Users/soler/Downloads/OrionA_StructureFunction.png', bbox_inches='tight')
   plt.close()
   import pdb; pdb.set_trace() 

#exampleVisualization();
exampleHRO2D();
#exampleVisualization();


