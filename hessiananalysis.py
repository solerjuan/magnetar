# 
#
# Copyright (C) 2020 Juan Diego Soler

import sys
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.stats import circmean, circstd

# --------------------------------------------------------------------------------------------------------
def HessianAnalysis(inmap, pxksz=3, mode='reflect', nruns=1, s_inmap=None, mask=None):

   if np.logical_or(s_inmap is None, nruns<2):
    
      output=HessianAnalysisLITE(inmap, pxksz=pxksz, mode=mode)
      return {'lplus':output['lplus'], 's_lplus': np.nan, 'lminus':output['lminus'], 's_lminus': np.nan, 'theta':output['theta'], 's_theta':np.nan, 'sima':output['sima']}

   else:

      sz=np.shape(inmap)
      arrlplus =np.zeros([nruns,sz[0],sz[1]])
      arrlminus=np.zeros([nruns,sz[0],sz[1]])
      arrtheta =np.zeros([nruns,sz[0],sz[1]])

      for i in range(0,nruns):

         inmaprand=np.random.normal(loc=inmap, scale=s_inmap+0.*inmap)
         output=HessianAnalysisLITE(inmaprand, pxksz=pxksz, mode=mode)
 
         arrlplus[i]=output['lplus']       
         arrlminus[i]=output['lminus']
         arrtheta[i]=output['theta']
         sima=output['sima']       
 
      theta=circmean(arrtheta, axis=0, low=-np.pi/2, high=np.pi/2)
      s_theta=circstd(arrtheta, axis=0, low=-np.pi/2, high=np.pi/2)
      lminus=np.mean(arrlminus, axis=0)
      s_lminus=np.std(arrlminus, axis=0)
      lplus=np.mean(arrlplus, axis=0)
      s_lplus=np.std(arrlplus, axis=0)

      return {'lplus':lplus, 's_lplus':s_lplus, 'lminus':lminus, 's_lminus':s_lminus, 'theta':theta, 's_theta':s_theta, 'sima':sima}


# --------------------------------------------------------------------------------------------------------
def HessianAnalysisLITE(inmap, pxksz=3, mode='reflect'):

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

   return {'lplus':lplus, 'lminus':lminus, 'theta':np.arctan(np.tan(theta)), 'sima': sima}


