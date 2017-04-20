# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

def anglemean(angle):

	vx=np.cos(angle)
	vy=np.sin(angle)

	meanangle=np.arctan2(np.mean(vy),np.mean(vx))

	return meanangle



def structurefunction(Qmap, Umap, lag=1.0, s_lag=0.5, mask=0.0, pitch=1.0):

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

hdu=fits.open('data/Taurusfwhm5_logNHmap.fits')
Imap0=hdu[0].data
hdu=fits.open('data/Taurusfwhm10_Qmap.fits')
Qmap0=hdu[0].data
hdu=fits.open('data/Taurusfwhm10_Umap.fits')
Umap0=hdu[0].data
#mask=0*Imap[0].data
#mask[(Imap[0].data > 21.5).nonzero()]=1
#structurefunction(Qmap[0].data, Umap[0].data, lag=5.0, s_lag=0.5, pitch=1.0)

#Qmap=np.random.uniform(-1., 1., (64,64))
#Umap=np.random.uniform(-1., 1., (64,64))
Imap=Imap0#[100:400,100:400]
Qmap=Qmap0#[100:400,100:400]
Umap=Umap0#[100:400,100:400]
mask=1+0*Qmap
mask[(Imap < 21.5).nonzero()]=0

plt.imshow(Imap*mask, origin='lower', clim=(21.5, np.max(Imap)))
plt.show()

lags=np.arange(2.,40.,4.)
s2=np.zeros(np.size(lags))

for i in range(0, np.size(lags)):	
	s2[i]=structurefunction(Qmap, Umap, mask=mask, lag=lags[i], s_lag=1., pitch=1.)

plt.plot(lags, s2, color='red')
plt.show()


