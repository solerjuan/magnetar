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

from scipy import interpolate

# ================================================================================================================================
def planckct():

   colombi1_cmap = matplotlib.colors.ListedColormap(np.loadtxt("Planck_Parchment_RGB.txt")/255.)
   colombi1_cmap.set_bad("gray") # color of missing pixels
   colombi1_cmap.set_under("blue")

   return colombi1_cmap

# ================================================================================================================================
def lic(vx, vy, length=8, niter=1, normalize=True, amplitude=False, level=0.1, scalar=1, interpolation='nearest'):
   # Calculates the line integral convolution representation of the 2D vector field represented by Vx and Vy.
   # INPUTS
   # Vx     - X
   # Vy     - Y
   # length - L

   vxbad=np.isnan(vx).nonzero()
   vybad=np.isnan(vy).nonzero()

   vx[vxbad]=0. 
   vy[vybad]=0.

   sz=np.shape(vx)

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

   vl=np.random.rand(sz[0],sz[1])

   xj=np.arange(sz[0])
   xi=np.arange(sz[1])

   for i in range(0,niter):

      texture=vl
      vv=np.zeros(sz)

      pi0, pj0 = np.meshgrid(xi, xj)
      pi, pj = np.meshgrid(xi, xj)
      mi=pi 
      mj=pj
        
      ppi=1.*pi
      ppj=1.*pj
      mmi=1.*mi
      mmj=1.*mj

      for l in range(0,length):
         print(l)

         ppi0=ppi
         ppj0=ppj
         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([ppi.ravel(),ppj.ravel()]))
         dpi=interpolate.griddata(points, ux.ravel(), outpoints, method=interpolation)
         dpj=interpolate.griddata(points, uy.ravel(), outpoints, method=interpolation)

         ppi=ppi0+0.25*np.reshape(dpi,[sz[0],sz[1]])
         ppj=ppj0+0.25*np.reshape(dpj,[sz[0],sz[1]])

         mmi0=mmi
         mmj0=mmj
         points   =np.transpose(np.array([pi0.ravel(),pj0.ravel()]))
         outpoints=np.transpose(np.array([mmi.ravel(),mmj.ravel()]))
         dmi=interpolate.griddata(points, ux.ravel(), outpoints, method=interpolation)
         dmj=interpolate.griddata(points, uy.ravel(), outpoints, method=interpolation)

         mmi=mmi0-0.25*np.reshape(dmi,[sz[0],sz[1]])
         mmj=mmj0-0.25*np.reshape(dmj,[sz[0],sz[1]])

         pi=(np.fix(ppi) + sz[0]) % sz[0]
         pj=(np.fix(ppj) + sz[1]) % sz[1]
         mi=(np.fix(mmi) + sz[0]) % sz[0]
         mj=(np.fix(mmj) + sz[1]) % sz[1]

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
    
         vv=vv.copy() + np.reshape(tempA,[sz[0],sz[1]]) + np.reshape(tempB,[sz[0],sz[1]])

      vl=0.25*vv/length
      #import pdb; pdb.set_trace()     

   vl[vxbad]=np.nan
   vl[vybad]=np.nan

   return vl


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

   X, Y = np.meshgrid(np.arange(0, sz[1]-1, pitch), np.arange(0, sz[0]-1, pitch))
   ux0=ux[Y,X]
   uy0=uy[Y,X]
   
   return X, Y, ux0, uy0



