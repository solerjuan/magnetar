# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel
from scipy import ndimage

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from reproject import reproject_interp

from bvisual import *

# -----------------------------------------------------------------------------------------------------------
def findclosedcontour(Imap, bins=100):
 
   minI=np.nan
   sz=np.shape(Imap)

   good=np.isfinite(Imap).nonzero()
   hist, bin_edges = np.histogram(Imap[good], bins=bins, density=True, range=[np.percentile(Imap[good], 5.), np.max(Imap[good])])
   bin_centres=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])   

   tempImap=Imap.copy()
   tempImap[np.isnan(tempImap).nonzero()]=0. #np.min(Imap[good]) 

   for i in range(0, bins):
      positions=np.logical_and(tempImap>=bin_edges[i],tempImap<=bin_edges[i+1]).nonzero()   

      if (np.max(positions[0]+1) < sz[0]):
         if (np.size(np.isnan(Imap[positions[0]+1,positions[1]]).nonzero()) == 0):
            if (np.min(positions[0]-1) > 0): 
               if (np.size(np.isnan(Imap[positions[0]-1,positions[1]]).nonzero()) == 0):
                  if (np.max(positions[1]+1) < sz[1]):
                     if (np.size(np.isnan(Imap[positions[0],positions[1]+1]).nonzero()) == 0):
                        if (np.min(positions[1]-1) > 0):
                           if (np.size(np.isnan(Imap[positions[0]+1,positions[1]-1]).nonzero()) == 0):
                              minI=bin_edges[i]
                              break

   return minI


# -----------------------------------------------------------------------------------------------------------
def gradpoverp(Qmap, Umap, ksz=1, mode='nearest'):

   sQmap=ndimage.filters.gaussian_filter(Qmap, [ksz, ksz], order=[0,0], mode=mode)
   dQdx =ndimage.filters.gaussian_filter(Qmap, [ksz, ksz], order=[0,1], mode=mode)
   dQdy =ndimage.filters.gaussian_filter(Qmap, [ksz, ksz], order=[1,0], mode=mode)
   normGradQ=np.sqrt(dQdx**2+dQdy**2)

   sUmap=ndimage.filters.gaussian_filter(Umap, [ksz, ksz], order=[0,0], mode=mode)
   dUdx =ndimage.filters.gaussian_filter(Umap, [ksz, ksz], order=[0,1], mode=mode)
   dUdy =ndimage.filters.gaussian_filter(Umap, [ksz, ksz], order=[1,0], mode=mode)
   normGradU=np.sqrt(dUdx**2+dUdy**2)

   P=np.sqrt(Qmap**2+Umap**2)   
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

   meanq=np.mean(stokesq)
   meanu=np.mean(stokesu)
   meanpsi=0.5*np.arctan2(meanu,meanq) 
 
   return meanpsi

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
   p_gal=np.sqrt(Qmap**2+Umap**2)

   #ex_gal=-np.sin(psi_gal)
   #ey_gal=np.cos(psi_gal)

   #hduIN=fits.PrimaryHDU(ex_gal)
   #hduIN.header=header
   #TEMPex_gal, footprint = reproject_interp(hduIN, hdrOUT) 
   #hduIN=fits.PrimaryHDU(ey_gal)
   #hduIN.header=header
   #TEMPey_gal, footprint = reproject_interp(hduIN, hdrOUT)

   hduIN=fits.PrimaryHDU(Qmap)
   hduIN.header=header
   TEMPqmap, footprint = reproject_interp(hduIN, hdrOUT)         
   hduIN=fits.PrimaryHDU(Umap)
   hduIN.header=header
   TEMPumap, footprint = reproject_interp(hduIN, hdrOUT)

   TEMPp_gal=np.sqrt(TEMPqmap**2+TEMPumap**2)
   TEMPpsi_gal=0.5*np.arctan2(-TEMPumap,TEMPqmap) 
   TEMPex_gal=-1.*TEMPp_gal*np.sin(TEMPpsi_gal)
   TEMPey_gal=    TEMPp_gal*np.cos(TEMPpsi_gal)

   galnorth_gal=SkyCoord(l=0.*u.degree, b=90.*u.degree, frame='galactic')
   galnorth_equ=galnorth_gal.transform_to('fk5')
   alphaGN=galnorth_equ.ra.value
   deltaGN=galnorth_equ.dec.value

   angc=np.arccos(np.sin(deltaGN*np.pi/180.0))
     
   alpha0, delta0=np.meshgrid(ra, dec) 

   angb=np.arccos(np.sin(delta0*np.pi/180.0))
   anga=np.arccos(np.sin(deltaGN*np.pi/180.0)*np.sin(delta0*np.pi/180.0)+np.cos(deltaGN*np.pi/180.0)*np.cos(delta0*np.pi/180.0)*np.cos((alpha0-alphaGN)*np.pi/180.0))

   alpha=np.arccos((np.cos(angc)-np.cos(anga)*np.cos(angb))/(np.sin(anga)*np.sin(angb)))

   ex_equ=np.cos(alpha)*TEMPex_gal-np.sin(alpha)*TEMPey_gal
   ey_equ=np.sin(alpha)*TEMPex_gal+np.cos(alpha)*TEMPey_gal

   Qmap_equ=ey_equ**2-ex_equ**2
   Umap_equ=2.*ey_equ*ex_equ

   return Imap_equ, Qmap_equ, Umap_equ, hdrOUT

   #import pdb; pdb.set_trace()

