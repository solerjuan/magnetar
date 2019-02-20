# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from reproject import reproject_interp

from bvisual import *

# -----------------------------------------------------------------------------------------------------------
def gradpoverp(Qmap, Umap, ksz=1):

   if (ksz <= 1):
      sQmap=Qmap
      sUmap=Umap
   else:
      sQmap=convolve_fft(Qmap, Gaussian2DKernel(float(pxksz)))
      sUmap=convolve_fft(Umap, Gaussian2DKernel(float(pxksz)))

   P=np.sqrt(sQmap**2 + sUmap**2)
  
   gradQ=np.gradient(Qmap)
   normGradQ=np.sqrt(gradQ[0]**2 + gradQ[1]**2)
   dQdx=gradQ[0]
   dQdy=gradQ[1]

   gradU=np.gradient(Umap)
   normGradU=np.sqrt(gradU[0]**2 + gradU[1]**2)
   dUdx=gradU[0]
   dUdy=gradU[1]

   gradP=np.sqrt(dQdx**2 + dQdy**2 + dUdx**2 +dUdy**2)

   nopol=np.logical_or(normGradQ==0.,normGradU==0.).nonzero()

   gradP[nopol]=0. 
   P[nopol]=0.

   gradPoverP=gradP/P 

   return gradPoverP

# -----------------------------------------------------------------------------------------------------------
def anglediff(angle1, angle2):

   phi=np.arctan2(np.tan(angle1)-np.tan(angle2),1+np.tan(angle1)*np.tan(angle2))
  
   return phi

# -----------------------------------------------------------------------------------------------------------
def anglemean(stokesq, stokesu):

   return 0

# -----------------------------------------------------------------------------------------------------------
def angledispqu(stokesq, stokesu):

   meanq=np.mean(stokesq)
   meanu=np.mean(stokesu)

   meanpsi=0.5*np.arctan2(meanu,meanq)
   deltapsi=0.5*np.arctan2(stokesq*meanu-meanq*stokesu, stokesq*meanq+meanu*stokesu)
   disp=np.sqrt(np.mean(deltapsi**2))
   return disp 

# ------------------------------------------------------------------------------------------------------------
def polgal2equ(Imap, Qmap, Umap, header):

   glon=header['CRVAL1']+header['CDELT1']*(np.arange(header['NAXIS1'])-header['CRPIX1'])
   glat=header['CRVAL2']+header['CDELT2']*(np.arange(header['NAXIS2'])-header['CRPIX2'])     

   c_gal=SkyCoord(l=np.mean(glon)*u.degree, b=np.mean(glat)*u.degree, frame='galactic')
   c_equ=c_gal.transform_to('fk5')

   hduOUT=fits.PrimaryHDU(Imap)
   hduOUT.header['CTYPE1']='RA---CAR'
   hduOUT.header['CTYPE2']='DEC--CAR'
   hduOUT.header['CRPIX1']=hduOUT.header['NAXIS1']/2.
   hduOUT.header['CRPIX2']=hduOUT.header['NAXIS2']/2.
   hduOUT.header['CRVAL1']=c_equ.ra.value
   hduOUT.header['CRVAL2']=c_equ.dec.value
   hduOUT.header['CDELT1']=header['CDELT1']
   hduOUT.header['CDELT2']=header['CDELT2']
   hdrOUT=hduOUT.header

   ra=hdrOUT['CRVAL1']+hdrOUT['CDELT1']*(np.arange(hdrOUT['NAXIS1'])-hdrOUT['CRPIX1'])
   dec=hdrOUT['CRVAL2']+hdrOUT['CDELT2']*(np.arange(hdrOUT['NAXIS2'])-hdrOUT['CRPIX2'])

   hduIN=fits.PrimaryHDU(Imap)
   hduIN.header=header
   Imap_equ, footprintI = reproject_interp(hduIN, hdrOUT)
 
   psi_gal=0.5*np.arctan2(-Umap,Qmap)
   ex_gal=-np.sin(psi_gal)
   ey_gal=np.cos(psi_gal)

   bx=ey_gal
   by=-ex_gal
   x, y, ux, uy = vectors(Imap, bx, by, pitch=25)

   ax1=plt.subplot(1,1,1, projection=WCS(hdrREF))
   im=ax1.imshow(Imap, origin='lower', cmap=planckct())
   arrows=plt.quiver(x, y, ux, uy, units='width', color='black', pivot='middle', headlength=0, headwidth=0)
   plt.colorbar(im)
   plt.show()

   hduIN=fits.PrimaryHDU(ex_gal)
   hduIN.header=header
   TEMPex_gal, footprint = reproject_interp(hduIN, hdrOUT) 
   hduIN=fits.PrimaryHDU(ey_gal)
   hduIN.header=header
   TEMPey_gal, footprint = reproject_interp(hduIN, hdrOUT)

   galnorth_gal=SkyCoord(l=0.*u.degree, b=90.*u.degree, frame='galactic')
   galnorth_equ=galnorth_gal.transform_to('fk5')
   alphaGN=galnorth_equ.ra.value
   deltaGN=galnorth_equ.dec.value

   angc=np.arccos(np.sin(deltaGN*np.pi/180.0))
     
   alpha0, delta0=np.meshgrid(ra, dec) 
   #alpha0=np.mean(ra)
   #delta0=np.mean(dec)

   angb=np.arccos(np.sin(delta0*np.pi/180.0))
   anga=np.arccos(np.sin(deltaGN*np.pi/180.0)*np.sin(delta0*np.pi/180.0)+np.cos(deltaGN*np.pi/180.0)*np.cos(delta0*np.pi/180.0)*np.cos((alpha0-alphaGN)*np.pi/180.0))

   alpha=np.arccos((np.cos(angc)-np.cos(anga)*np.cos(angb))/(np.sin(anga)*np.sin(angb)))

   ex_equ=np.cos(alpha)*TEMPex_gal-np.sin(alpha)*TEMPey_gal
   ey_equ=np.sin(alpha)*TEMPex_gal+np.cos(alpha)*TEMPey_gal

   Qmap_equ=ey_equ**2-ex_equ**2
   Umap_equ=2.*ey_equ*ex_equ      

   return Imap_equ, Qmap_equ, Umap_equ, hdrOUT

   #import pdb; pdb.set_trace()

indir='data/'
hdu=fits.open(indir+'Taurusfwhm5_logNHmap.fits')
Imap=hdu[0].data
hdrREF=hdu[0].header
hdu=fits.open(indir+'Taurusfwhm10_Qmap.fits')
Qmap=hdu[0].data
hdu=fits.open(indir+'Taurusfwhm10_Umap.fits')
Umap=hdu[0].data

Imap_equ, Qmap_equ, Umap_equ, hdrOUT=polgal2equ(Imap, Qmap, Umap, hdrREF)

psi=0.5*np.arctan2(Umap_equ,Qmap_equ)
ex=np.sin(psi)
ey=np.cos(psi)
bx=ey
by=-ex
x, y, ux, uy = vectors(Imap_equ, bx, by, pitch=25)

ax1=plt.subplot(1,1,1, projection=WCS(hdrOUT))
im=ax1.imshow(Imap_equ, origin='lower', cmap=planckct())
arrows=plt.quiver(x, y, ux, uy, units='width', color='black', pivot='middle', headlength=0, headwidth=0)
plt.colorbar(im)
plt.show()


