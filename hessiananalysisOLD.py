# 
#
# Copyright (C) 2020 Juan Diego Soler

import sys
import numpy as np
from scipy import ndimage
from scipy import signal

# --------------------------------------------------------------------------------------------------------
def HessianAnalysis(inmap, pxksz=3, mode='reflect'):

   # pxksz - Standard deviation of the derivative kernel - fwhm/2.355

   sima=ndimage.filters.gaussian_filter(inmap, [pxksz, pxksz], order=[0,0], mode=mode)

   dIdx=ndimage.filters.gaussian_filter(inmap, [pxksz, pxksz], order=[0,1], mode=mode)
   dIdy=ndimage.filters.gaussian_filter(inmap, [pxksz, pxksz], order=[1,0], mode=mode)
 
   Hxx=ndimage.filters.gaussian_filter(dIdx, [pxksz, pxksz], order=[0,1], mode=mode)
   Hyx=ndimage.filters.gaussian_filter(dIdy, [pxksz, pxksz], order=[0,1], mode=mode)
   Hxy=ndimage.filters.gaussian_filter(dIdx, [pxksz, pxksz], order=[1,0], mode=mode)
   Hyy=ndimage.filters.gaussian_filter(dIdy, [pxksz, pxksz], order=[1,0], mode=mode)

   lplus = 0.5*(Hxx+Hyy)+0.5*np.sqrt((Hxx-Hyy)**2+4.*Hxy*Hyx)
   lminus= 0.5*(Hxx+Hyy)-0.5*np.sqrt((Hxx-Hyy)**2+4.*Hxy*Hyx)
   theta=0.5*np.arctan2(Hxy+Hyx, Hxx-Hyy)

   return {'lplus':lplus, 'lminus':lminus, 'theta':theta, 'sima': sima}


