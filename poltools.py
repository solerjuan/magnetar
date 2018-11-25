# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

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

