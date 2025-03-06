# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2019 Juan Diego Soler

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

from astropy.stats.circstats import *

from nose.tools import assert_equal, assert_true

from scipy.ndimage import gaussian_filter

from bvisual import *
from poltools import *
from astropy.wcs import WCS

from tqdm import tqdm

# # ----------------------------------------------------------------------------------------
def s2l(Qmap, Umap, lags=[4.0], s_lag=1.0, mask=None, header=None):

   assert Qmap.shape == Umap.shape, "Dimensions of Qmap and Umap must match"

   sz=np.shape(Qmap)
   QmapRAND=(np.random.rand(sz[0],sz[1])-0.5)
   UmapRAND=(np.random.rand(sz[0],sz[1])-0.5)

   if mask is None:
      mask=np.ones_like(Qmap)
   assert Qmap.shape == mask.shape, "Dimensions of mask and Qmap must match"
   Qmap[(mask==0.).nonzero()]=np.nan
   Umap[(mask==0.).nonzero()]=np.nan
   
   # ------------------------------------------------------------------------------------------------
   if header is None:
      posx, posy=np.meshgrid(np.arange(0,sz[0]), np.arange(0,sz[1]))
   else:
      ra=header['CRVAL1']+header['CDELT1']*(np.arange(header['NAXIS1'])-header['CRPIX1'])
      dec=header['CRVAL2']+header['CDELT2']*(np.arange(header['NAXIS2'])-header['CRPIX2'])
      posx, posy=np.meshgrid(ra, dec)

   # ----------------------------------------------------------------------
   dist=np.zeros(np.size(posx)*np.size(posx))

   for i in tqdm(range(0,np.size(posx))):
      tempdeltax=posx.ravel()[i]-posx.ravel()
      tempdeltay=posy.ravel()[i]-posy.ravel()
      dist[i*np.size(posx):(i+1)*np.size(posx)]=np.sqrt(tempdeltax**2+tempdeltay**2)
   posx=None
   posy=None

   # ----------------------------------------------------------------------------------------------
   stwo=np.nan
   stwoRAND=np.nan
   stwoarr=np.nan*np.zeros_like(lags)
   stwoRANDarr=np.nan*np.zeros_like(lags)   
   ngood=np.nan*np.zeros_like(lags)

   dhist, bin_edges=np.histogram(dist, bins=100)
   bin_centers=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
 
   for i in range(0,np.size(lags)):
      print('Lag between', lags[i]-s_lag, 'and', lags[i]+s_lag)
      #good=(np.abs(dist-lags[i]) < s_lag).nonzero()
      good = np.logical_and(dist >= lags[i]-s_lag, dist < lags[i]+s_lag).nonzero()
      #good1, good2 = np.logical_and(dist >= lags[i]-s_lag, dist < lags[i]+s_lag).nonzero()

      good1=good[0]%(sz[0]*sz[0]);
      good2=(good[0]/(sz[0]*sz[0])).astype(int);

      maskvec=mask.ravel()
      Qvec=Qmap.ravel(); Qvec[maskvec > 1.]=np.nan
      Uvec=Umap.ravel(); Uvec[maskvec > 1.]=np.nan
      QvecRAND=QmapRAND.ravel(); QvecRAND[maskvec > 1.]=np.nan
      UvecRAND=UmapRAND.ravel(); UvecRAND[maskvec > 1.]=np.nan 

      if np.logical_and(np.size(good1) > 0, np.size(good2) > 0):
          # --------------------------------------------------------------------------------
         Q1=Qvec[good1]
         U1=Uvec[good1]
         Q2=Qvec[good2]
         U2=Uvec[good2]
         good=np.logical_and(np.logical_and(np.isfinite(Q1),np.isfinite(Q2)),np.logical_and(np.isfinite(U1),np.isfinite(U2))).nonzero()
         # From Eq. (11) in Planck XIX. A&A 576 (2015) A104
         deltapsi=0.5*np.arctan2(Q1[good]*U2[good]-Q2[good]*U1[good], Q1[good]*Q2[good]+U1[good]*U2[good])

         # From Eq. (6) in Planck XIX. A&A 576 (2015) A104
         gooddeltapsi=deltapsi[np.isfinite(deltapsi).nonzero()]
         weights=0.5*np.ones_like(gooddeltapsi)
         stwo=np.sqrt(np.sum(weights*gooddeltapsi**2)/np.sum(weights))

         # --------------------------------------------------------------------------------
         Q1=QvecRAND[good1]
         U1=UvecRAND[good1]
         Q2=QvecRAND[good2]
         U2=UvecRAND[good2]
         good=np.logical_and(np.logical_and(np.isfinite(Q1),np.isfinite(Q2)),np.logical_and(np.isfinite(U1),np.isfinite(U2))).nonzero()
         # From Eq. (11) in Planck XIX. A&A 576 (2015) A104
         deltapsiRAND=0.5*np.arctan2(Q1[good]*U2[good]-Q2[good]*U1[good], Q1[good]*Q2[good]+U1[good]*U2[good])

         # From Eq. (6) in Planck XIX. A&A 576 (2015) A104
         gooddeltapsiRAND=deltapsiRAND[np.isfinite(deltapsiRAND).nonzero()]
         weights=0.5*np.ones_like(gooddeltapsiRAND)
         stwoRAND=np.sqrt(np.sum(weights*gooddeltapsiRAND**2)/np.sum(weights))

      else:
         print("No points in the selected lag range")
         stwo=np.nan
         stwoRAND=np.nan

      ngood[i]=np.size(good)
      stwoarr[i]=stwo
      stwoRANDarr[i]=stwoRAND

   return {'S2': stwoarr, 'S2rand': stwoRANDarr, 'npairs': ngood}

# ----------------------------------------------------------------------------------------
def s2(Qmap, Umap, lags=[4.0], s_lag=1.0, mask=None, header=None):

   assert Qmap.shape == Umap.shape, "Dimensions of Qmap and Umap must match"

   sz=np.shape(Qmap)
   QmapRAND=(np.random.rand(sz[0],sz[1])-0.5)
   UmapRAND=(np.random.rand(sz[0],sz[1])-0.5)

   if mask is None:
      mask=np.ones_like(Qmap)
   assert Qmap.shape == mask.shape, "Dimensions of mask and Qmap must match"
   Qmap[(mask==0.).nonzero()]=np.nan
   Umap[(mask==0.).nonzero()]=np.nan

   # ------------------------------------------------------------------------------------------------
   if header is None:
      posx, posy=np.meshgrid(np.arange(0,sz[0]), np.arange(0,sz[1]))
   else:
      ra=header['CRVAL1']+header['CDELT1']*(np.arange(header['NAXIS1'])-header['CRPIX1'])
      dec=header['CRVAL2']+header['CDELT2']*(np.arange(header['NAXIS2'])-header['CRPIX2'])
      posx, posy=np.meshgrid(ra, dec)

   print('Calculation of x positions')
   x1, x2 =np.meshgrid(posx.ravel(), posx.ravel())
   deltax=(x1-x2)
   x1=None; x2=None
   print('Calculation of y positions')
   y1, y2 =np.meshgrid(posy.ravel(), posy.ravel())
   deltay=(y1-y2)
   y1=None; y2=None
   print('Calculation of distances')
   dist=np.sqrt(deltax**2+deltay**2)
   deltax=None; deltay=None;

   # ----------------------------------------------------------------------------------------------
   stwo=np.nan
   stwoRAND=np.nan
   stwoarr=np.nan*np.zeros_like(lags)
   stwoRANDarr=np.nan*np.zeros_like(lags)  

   for i in range(0,np.size(lags)):
      print('Lag between', lags[i]-s_lag, 'and', lags[i]+s_lag)
      good1, good2 = np.logical_and(dist >= lags[i]-s_lag, dist < lags[i]+s_lag).nonzero()

      maskvec=mask.ravel() 
      Qvec=Qmap.ravel(); Qvec[maskvec > 1.]=np.nan
      Uvec=Umap.ravel(); Uvec[maskvec > 1.]=np.nan
      QvecRAND=QmapRAND.ravel(); QvecRAND[maskvec > 1.]=np.nan
      UvecRAND=UmapRAND.ravel(); UvecRAND[maskvec > 1.]=np.nan  

      if np.logical_and(np.size(good1) > 0, np.size(good2) > 0):
          # --------------------------------------------------------------------------------
         Q1=Qvec[good1]
         U1=Uvec[good1]
         Q2=Qvec[good2]
         U2=Uvec[good2]
         good=np.logical_and(np.logical_and(np.isfinite(Q1),np.isfinite(Q2)),np.logical_and(np.isfinite(U1),np.isfinite(U2))).nonzero()
         # From Eq. (11) in Planck XIX. A&A 576 (2015) A104
         deltapsi=0.5*np.arctan2(Q1[good]*U2[good]-Q2[good]*U1[good], Q1[good]*Q2[good]+U1[good]*U2[good])
    
         # From Eq. (6) in Planck XIX. A&A 576 (2015) A104
         gooddeltapsi=deltapsi[np.isfinite(deltapsi).nonzero()]
         weights=0.5*np.ones_like(gooddeltapsi)
         stwo=np.sqrt(np.sum(weights*gooddeltapsi**2)/np.sum(weights))

         # --------------------------------------------------------------------------------
         Q1=QvecRAND[good1]
         U1=UvecRAND[good1]
         Q2=QvecRAND[good2]
         U2=UvecRAND[good2]
         good=np.logical_and(np.logical_and(np.isfinite(Q1),np.isfinite(Q2)),np.logical_and(np.isfinite(U1),np.isfinite(U2))).nonzero()
         # From Eq. (11) in Planck XIX. A&A 576 (2015) A104
         deltapsiRAND=0.5*np.arctan2(Q1[good]*U2[good]-Q2[good]*U1[good], Q1[good]*Q2[good]+U1[good]*U2[good])

         # From Eq. (6) in Planck XIX. A&A 576 (2015) A104
         gooddeltapsiRAND=deltapsiRAND[np.isfinite(deltapsiRAND).nonzero()]
         weights=0.5*np.ones_like(gooddeltapsiRAND)
         stwoRAND=np.sqrt(np.sum(weights*gooddeltapsiRAND**2)/np.sum(weights))

      else:
         print("No points in the selected lag range")
         stwo=np.nan 
         stwoRAND=np.nan

      stwoarr[i]=stwo
      stwoRANDarr[i]=stwoRAND
      #import pdb; pdb.set_trace()

   return stwoarr, stwoRANDarr

# ------------------------------------------------------------------------------------------------------------------------------
def structurefunction(Qmap, Umap, lag=4.0, s_lag=1.0, mask=None, header=None):
   """ Calculates the spatial correlation between im1 and im2 using the HOG method 

   Parameters
   ----------   
    Qmap : array corresponding to the first image to be compared 
    Umap : array corresponding to the second image to be compared
   
   Returns
   -------
    hogcorr :  
    corrframe :

   Examples
   --------
   """
   assert Qmap.shape == Umap.shape, "Dimensions of Qmap and Umap must match"

   if mask is None:
      mask=np.ones_like(Qmap)
   assert Qmap.shape == mask.shape, "Dimensions of mask and Qmap must match"
   Qmap[(mask==0.).nonzero()]=np.nan
   Umap[(mask==0.).nonzero()]=np.nan

   sz=np.shape(Qmap)

   if header is None:
      posx, posy = np.meshgrid(np.arange(0,sz[0]), np.arange(0,sz[1])) 
   else:
      ra=header['CRVAL1']+header['CDELT1']*(np.arange(header['NAXIS1'])-header['CRPIX1'])
      dec=header['CRVAL2']+header['CDELT2']*(np.arange(header['NAXIS2'])-header['CRPIX2'])
      posx, posy=np.meshgrid(ra, dec)

   x1, x2 =np.meshgrid(posx.ravel(), posx.ravel()) 
   deltax=(x1-x2)
   x1=None; x2=None
   y1, y2 =np.meshgrid(posy.ravel(), posy.ravel())    
   deltay=(y1-y2)
   y1=None; y2=None
   print('Calculation of distances')
   dist=np.sqrt(deltax**2+deltay**2)
   deltax=None; deltay=None;

   good1, good2 = np.logical_and(dist >= lag-s_lag, dist < lag+s_lag).nonzero()

   stwo=np.nan
   ngood=0.

   maskvec=mask.ravel() 
   Qvec=Qmap.ravel(); Qvec[maskvec > 1.]=np.nan
   Uvec=Umap.ravel(); Uvec[maskvec > 1.]=np.nan
   # From Eq. (11) in Planck XIX. A&A 576 (2015) A104
   if np.logical_and(np.size(good1) > 0, np.size(good2) > 0):

      Q1=Qvec[good1]
      U1=Uvec[good1]
      Q2=Qvec[good2]
      U2=Uvec[good2]
      good=np.logical_and(np.logical_and(np.isfinite(Q1),np.isfinite(Q2)),np.logical_and(np.isfinite(U1),np.isfinite(U2))).nonzero()
      deltapsi=0.5*np.arctan2(Q1[good]*U2[good]-Q2[good]*U1[good], Q1[good]*Q2[good]+U1[good]*U2[good])
      ngood=np.size(good)
      #deltapsi=0.5*np.arctan2(Qvec[good1]*Uvec[good2]-Qvec[good2]*Uvec[good1], Qvec[good1]*Qvec[good2]+Uvec[good1]*Uvec[good2])
      #import pdb; pdb.set_trace()

      # From Eq. (6) in Planck XIX. A&A 576 (2015) A104
      gooddeltapsi=deltapsi[np.isfinite(deltapsi).nonzero()]
      weights=0.5*np.ones_like(gooddeltapsi)   
      #stwo=np.std(deltapsi)
      stwo=np.sqrt(np.sum(weights*gooddeltapsi**2)/np.sum(weights))

   else:
      print("No points in the selected lag range")

   #print(stwo)
   from scipy.stats import circstd
   #print(circstd)
   #import pdb; pdb.set_trace()

   #return stwo
   return {'S2': stwo, 'npairs': ngood}  

# ------------------------------------------------------------------------------------------------------------------------------
def OLDstructurefunction(Qmap, Umap, lag=1.0, s_lag=0.5, mask=0.0, pitch=1.0):

   sz=np.shape(Qmap)
   x=np.arange(-sz[1]/2., sz[1]/2., 1.)
   y=np.arange(-sz[0]/2., sz[0]/2., 1.)
   xx, yy = np.meshgrid(x, y)

   #index=np.arange(np.size(xx))

   sfmap=0.*Qmap

   valid=(mask > 0).nonzero()

   for i in range(0, np.size(Qmap)):
      if (mask[np.unravel_index(i,sz)] > 0):

         diff=np.sqrt((xx[np.unravel_index(i,sz)]-xx)**2+(yy[np.unravel_index(i,sz)]-yy)**2)

         good=np.logical_and(mask>0, np.logical_and(diff>lag-s_lag,diff<lag+s_lag)).nonzero()
         #Qdiff=np.mean(Qmap[np.unravel_index(i,sz)]-Qmap[good])
         #Udiff=np.mean(Umap[np.unravel_index(i,sz)]-Umap[good])
         #sfmap[np.unravel_index(i,sz)]=0.5*np.arctan2(-Udiff, Qdiff)
         Qdiff=Qmap[np.unravel_index(i,sz)]*Qmap[good]+Umap[good]*Umap[np.unravel_index(i,sz)]
         Udiff=Qmap[np.unravel_index(i,sz)]*Umap[good]-Qmap[good]*Umap[np.unravel_index(i,sz)]
         angles=0.5*np.arctan2(Udiff, Qdiff)
         sfmap[np.unravel_index(i,sz)]=np.sum(angles**2)/float(np.size(angles)-1)
         #print(sfmap[np.unravel_index(i,sz)])

   #plt.imshow(sfmap*180./np.pi, origin='lower')
   #plt.show()

   goodsfmap=sfmap[valid]
   deltapsi=anglemean(goodsfmap[np.isfinite(goodsfmap).nonzero()])*180./np.pi
   print(deltapsi)
   #import pdb; pdb.set_trace()	

   return deltapsi

