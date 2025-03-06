# 
#
# Copyright (C) 2020 Juan Diego Soler

import sys
import numpy as np
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel


def GaussianDerivative(ima, sigma, angle=0., mode='reflect'):

   kernel=Gaussian2DKernel(sigma,sigma)
   dxkernel=ndimage.rotate(np.gradient(kernel, axis=1), angle)
   dykernel=ndimage.rotate(np.gradient(kernel, axis=0), angle)
   
   sima=ndimage.convolve(ima, kernel, mode=mode)
   dIdx=ndimage.convolve(ima, dxkernel, mode=mode)
   dIdy=ndimage.convolve(ima, dykernel, mode=mode)

   return {'sima':sima, 'dx':dIdx, 'dy':dIdy}

