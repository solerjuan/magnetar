# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

from congrid import *

from scipy import interpolate
from tqdm import tqdm

# ================================================================================================================================
def planckct():

   colombi1_cmap = matplotlib.colors.ListedColormap(np.loadtxt('Planck_Parchment_RGB.txt')/255.)
   colombi1_cmap.set_bad('white') # color of missing pixels
   colombi1_cmap.set_under("blue")

   return colombi1_cmap

# ========================================================================================================================
def lic(vx0, vy0, length=8, niter=1, normalize=True, amplitude=False, level=0.1, scalar=1, interpolation='nearest', inputmap=None, factor=1.):
   # Calculates the line integral convolution representation of the 2D vector field represented by Vx and Vy.
   # INPUTS
   # Vx     - X
   # Vy     - Y
   # length - L
 
   # Check if the images match
   assert vx0.shape == vy0.shape, "Dimensions of ima2 and ima1 must match"
   sz=np.shape(vx0)
   
    # Identify bad pixels
   vxbad=np.isnan(vx0).nonzero()
   vybad=np.isnan(vy0).nonzero()

   vx0[vxbad]=0.
   vy0[vybad]=0.
 
   # ===============================================================================================
   if (factor==1.):
      vx=np.copy(vx0)
      vy=np.copy(vy0)
   else:
      print('[LIC] Warning: rescaling input maps')
      vx=congrid(vx0, np.array([int(factor*sz[0]),int(factor*sz[1])]), method='linear')
      vy=congrid(vy0, np.array([int(factor*sz[0]),int(factor*sz[1])]), method='linear')
  
   # Assert new shape 
   sz=np.shape(vx)

   ni=sz[0]
   nj=sz[1]

   uu=np.sqrt(vx**2+vy**2)
   ii=(uu == 0.).nonzero()

   if (np.size(ii) > 0):
      uu[ii]=1.0
   
   if (normalize):
      ux=vx/uu
      uy=vy/uu
   else: 
      ux=vx/np.max(uu)
      uy=vy/np.max(uu)

   if (inputmap is None):
      vl=np.random.rand(ni,nj)
   else:
      vl=inputmap

   xi=np.arange(ni)
   xj=np.arange(nj)

   outvl=np.zeros([niter,ni,nj])
   
   for i in range(0,niter):

      print('iter {:.0f} / {:.0f}'.format(i+1, niter))

      texture=vl
      vv=np.zeros([ni,nj])

      pi0, pj0 = np.meshgrid(xi, xj, indexing ='ij')
      pi, pj   = np.meshgrid(xi, xj, indexing ='ij')
      mi=pi 
      mj=pj
        
      ppi=1.*pi
      ppj=1.*pj
      mmi=1.*mi
      mmj=1.*mj

      pbar = tqdm(total=length)

      for l in range(0,length):

         ppi0=ppi
         ppj0=ppj
         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([ppi.ravel(),ppj.ravel()]))
         dpi=interpolate.griddata(points, uy.ravel(), outpoints, method=interpolation)
         dpj=interpolate.griddata(points, ux.ravel(), outpoints, method=interpolation)

         ppi=ppi0+0.25*np.reshape(dpi,[ni,nj])
         ppj=ppj0+0.25*np.reshape(dpj,[ni,nj])

         mmi0=mmi
         mmj0=mmj
         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([mmi.ravel(),mmj.ravel()]))
         dmi=interpolate.griddata(points, uy.ravel(), outpoints, method=interpolation)
         dmj=interpolate.griddata(points, ux.ravel(), outpoints, method=interpolation)

         mmi=mmi0-0.25*np.reshape(dmi,[ni,nj])
         mmj=mmj0-0.25*np.reshape(dmj,[ni,nj])

         pi=(np.fix(ppi) + ni) % ni
         pj=(np.fix(ppj) + nj) % nj
         mi=(np.fix(mmi) + ni) % ni
         mj=(np.fix(mmj) + nj) % nj

         ppi=pi+(ppi.copy()-np.fix(ppi.copy()))
         ppj=pj+(ppj.copy()-np.fix(ppj.copy()))
         mmi=mi+(mmi.copy()-np.fix(mmi.copy()))
         mmj=mj+(mmj.copy()-np.fix(mmj.copy()))

         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([ppi.ravel(),ppj.ravel()]))
         tempA=interpolate.griddata(points, texture.ravel(), outpoints, method=interpolation)
   
         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([mmi.ravel(),mmj.ravel()]))
         tempB=interpolate.griddata(points, texture.ravel(), outpoints, method=interpolation)

         vv=vv.copy() + np.reshape(tempA,[ni,nj]) + np.reshape(tempB,[ni,nj])

         pbar.update()

      pbar.close()
     
      vl=0.25*vv/length

      outvl[i,:,:]=vl

   vl[vxbad]=np.nan
   vl[vybad]=np.nan

   return outvl


# ================================================================================================================================
def vectors(image, vx, vy, pitch=10, normalize=True, cmap='binary', savefile=False):
   # Calculates the line integral convolution representation of the 2D vector field represented by Vx and Vy.
   # INPUTS
   # Vx     - X
   # Vy     - Y
   # pitch  - 

   sz=np.shape(image)

   nx=sz[0]
   ny=sz[1]

   uu=np.sqrt(vx**2+vy**2)
   ii=(uu == 0.).nonzero()

   if (np.size(ii) > 0):
      uu[ii]=1.0

   if (normalize):
      ux=vx/uu
      uy=vy/uu
   else:
      ux=vx/np.max(uu)
      uy=vy/np.max(uu)
 
   ux[ii]=0.
   uy[ii]=0.
 
   X, Y = np.meshgrid(np.arange(0, sz[1]-1, pitch), np.arange(0, sz[0]-1, pitch))
   ux0=ux[Y,X]
   uy0=uy[Y,X]
   
   return X, Y, ux0, uy0



