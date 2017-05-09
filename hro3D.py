# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel


def roangles3D(dens, Bx, By, Bz):
	# Calculates the relative orientation angle between the density structures and the magnetic field.
	# INPUTS
	# dens - regular cube with the values of density 
	# Bx   - regular cube with the values of x-component of the magnetic field
	# By   - regular cube with the values of y-component of the magnetic field
	# Bz   - regular cube with the values of z-component of the magnetic field
	#
	# OUTPUTS
	# cosphi - regular cube with the cosine of the relative orientation angle 
	#          between the density gradient and the magnetic field.

    	grad=np.gradient(dens, edge_order=2)
	gx=grad[1]; gy=grad[0]; gz=grad[2];

	normgrad=np.sqrt(gx*gx+gy*gy+gz*gz)
	normb=np.sqrt(Bx*Bx+By*By+Bz*Bz)

	zerograd=(normgrad==0.).nonzero()	
	zerob=(normb==0.).nonzero()

	normcross=np.sqrt((gy*Bz-gz*By)**2+(gx*Bz-gz*Bx)**2+(gx*By-gy*Bx)**2)
	normdot=gx*Bx+gy*By+gz*Bz	
	# Here I calculate the angle using atan2 to avoid the numerical problems of acos or asin
	phigrad=np.arctan2(normcross,normdot) 

	# The cosine of the angle between the iso-density and B is the sine of the angle between
	# the density gradient and B.	
	cosphi=np.sin(phigrad)
	cosphi[(normgrad == 0.).nonzero()]=-32768
	cosphi[(normb    == 0.).nonzero()]=-32768
	
    	return cosphi


def equibins(dens, steps=10, mind=0.):
	# Calculates the edges of (steps) bins with equal number of voxels.
	# INPUTS
        # dens   - regular cube with the values of density 
	# steps  - number of bins with equal number of voxels
	# mind   - minimun value of density to be considered in the calculation of the density bins
	# OUTPUTS
	# dsteps - edges of the density bins	


	sz=np.shape(dens)
	hist, bin_edges = np.histogram(dens[(dens > mind).nonzero()], bins=10*sz[0]*sz[1])
	bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

	chist=np.cumsum(hist)
	pitch=np.max(chist)/float(steps)
	
	hsteps=pitch*np.arange(0,steps+1,1)
        dsteps=np.zeros(steps+1)

	for i in range(0, np.size(dsteps)-1):	                
		good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
		dsteps[i]=np.min(bin_centre[good])
	dsteps[np.size(dsteps)-1]=np.max(dens)

	return dsteps

def roparameter(cosphi, hist, s_cosphi=0.125):
	# Calculate the relative orientation parameter
	# INPUTS
	# cosphi   - vector with the reference values for the histogram
	# hist     - histogram of relative orientations	
	# s_cosphi - range for the definitions of parallel (0 < cosphi < s_cosphi) or 
	#            perpendicular (1-s_cosphi < cosphi < 1).

	para=(np.abs(cosphi)>1.-s_cosphi).nonzero()
	perp=(np.abs(cosphi)<s_cosphi).nonzero()

	xi=(np.sum(hist[para])-np.sum(hist[perp]))/float(np.sum(hist[para])+np.sum(hist[perp]))

	return xi

def hro3D(dens, Bx, By, Bz, steps=10, hsize=21, mind=0, outh=[0,4,9]):

	cosphi=roangles3D(dens, Bx, By, Bz)
	dsteps=equibins(dens, steps=steps, mind=mind)

	hros=np.zeros([steps,hsize])
	cdens=np.zeros(steps)
	xi=np.zeros(steps)
	scube=0.*dens

	for i in range(0, np.size(dsteps)-1):
		good=np.logical_and(dens>dsteps[i],dens<dsteps[i+1]).nonzero()
		hist, bin_edges = np.histogram(cosphi[good], bins=hsize, range=(-1.,1.))	
		bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
		hros[i,:]=hist
		scube[good]=i
		cdens[i]=np.mean([dsteps[i],dsteps[i+1]])	
		xi[i]=roparameter(bin_centre, hist)

	outsteps=np.size(outh)
	color=iter(cm.cool(np.linspace(0, 1, outsteps)))	
	fig=plt.figure()
	for i in range(0, outsteps):
		c=next(color)
		labeltext=str(dsteps[outh[i]])+' < n < '+str(dsteps[outh[i]+1]) 
		plt.plot(bin_centre, hros[outh[i],:], '-', linewidth=2, c=c, label=labeltext) #drawstyle
	plt.xlabel(r'cos($\phi$)')	
	plt.legend()
	plt.show()

	fig=plt.figure()
	plt.plot(cdens, xi, '-', linewidth=2, c=c)
	plt.axhline(y=0., c='k', ls='--')	
	plt.xlabel(r'log$_{10}$ ($n_{\rm H}/$cm$^{-3}$)')
	plt.ylabel(r'$\zeta$')
	#plt.savefig(prefix + '-' + 'ROvsLogNH' + 'Thres' + "%d" % (thr) + '.png')
	plt.show()

        return hros, cdens, zeta


def main(args=None):

    	if res.prnt:
		print('Primes: {0}'.format(primes))

from astropy.convolution import Gaussian2DKernel
g2D=Gaussian2DKernel(10)
sz=np.shape(g2D)
dens=np.dstack([g2D]*sz[0])

grad=np.gradient(dens, edge_order=2)
#Bx=grad[1]
#By=grad[0]
#Bz=grad[2]
Bx=np.random.uniform(low=-1., high=1., size=np.shape(dens))
By=np.random.uniform(low=-1., high=1., size=np.shape(dens))
#Bz=np.random.uniform(low=-1., high=1., size=np.shape(dens))
#Bx=0.*dens
#By=0.*dens
Bz=0.*dens
hros, cdens, zeta = hro3D(dens, Bx, By, Bz, mind=np.mean(dens))

