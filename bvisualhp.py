# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2025 Juan Diego Soler

import os
import sys

import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors

from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

from congrid import *

import healpy as hp
from healpy.newvisufunc import projview
from reproject import reproject_from_healpix, reproject_to_healpix

from scipy import interpolate
from tqdm import tqdm

sys.path.append('/Users/soler/Documents/PYTHON/magnetar/')
from bvisual import planckct, lic

# =============================================================================================
def lichp(Imap, Qmap, Umap, l0=0., b0=0, reso=0.25, niter=3, slen=0.1, planck=True, filename='test'):

   suffix='_l'+str(l0)+'b'+str(b0)+'_LICreso'+str(reso)+'niter'+str(niter)+'slen'+str(slen)

   deltal=360.0
   deltab=180.0
   dummy=np.zeros((int(deltab/reso),int(deltal/reso)))
   hdu=fits.PrimaryHDU(dummy)
   hdu.header['CDELT1']=-reso
   hdu.header['CTYPE1']='GLON-CAR' #'GLON-TAN'
   hdu.header['CRVAL1']=l0
   hdu.header['CRPIX1']=0.5*hdu.header['NAXIS1']
   hdu.header['CDELT2']=reso
   hdu.header['CTYPE2']='GLAT-CAR' #'GLAT-TAN'
   hdu.header['CRVAL2']=b0
   hdu.header['CRPIX2']=0.5*hdu.header['NAXIS2']

   glonvec=hdu.header['CDELT1']*(np.arange(hdu.header['NAXIS1'])-hdu.header['CRPIX1'])+hdu.header['CRVAL1']
   glatvec=hdu.header['CDELT2']*(np.arange(hdu.header['NAXIS2'])-hdu.header['CRPIX2'])+hdu.header['CRVAL2']
   extent=np.array([np.max(glonvec),np.min(glonvec),np.min(glatvec),np.max(glatvec)])

   glonmat, glatmat = np.meshgrid(glonvec,glatvec)

   cartImap, footprint = reproject_from_healpix((Imap, 'galactic'), hdu.header, nested=False)
   cartQmap, footprint = reproject_from_healpix((Qmap, 'galactic'), hdu.header, nested=False)
   cartUmap, footprint = reproject_from_healpix((Umap, 'galactic'), hdu.header, nested=False)

   if (planck):
      psi=0.5*np.arctan2(-cartUmap,cartQmap)
      ex=np.sin(psi); ey=-np.cos(psi)
      bx=ey; by=-ex
   else:
      psi=0.5*np.arctan2(cartUmap,cartQmap)
      ex=np.cos(psi); ey=-np.sin(psi)
      bx=ey; by=-ex

   sz=np.shape(cartImap)
  
   # Compute LIC mask
   length=int(slen*sz[0])

   liccube=lic(bx, by, length=length, niter=niter)
   cartLICmap=liccube[niter-1]

   licmin=np.mean(cartLICmap)-np.std(cartLICmap)
   licmax=np.mean(cartLICmap)+np.std(cartLICmap)

   fig = plt.figure(figsize=(13.0,6.5))
   plt.rc('font', size=10)
   ax1=plt.subplot(111)
   ax1.pcolormesh(-np.deg2rad(glonmat), np.deg2rad(glatmat), cartImap, norm=colors.LogNorm(), cmap=planckct(), shading='nearest')
   ax1.pcolormesh(-np.deg2rad(glonmat), np.deg2rad(glatmat), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.3, shading='nearest')
   myxticks=ax1.get_xticks()[(np.abs(ax1.get_xticks()) < np.pi).nonzero()]
   xticks=[-180.,-90.,0.,90.,180.]
   ax1.set_xticks(np.deg2rad(xticks), labels=xticks)
   yticks=[-60.,-30.,0.,30.,60.]
   ax1.set_yticks(np.deg2rad(yticks), labels=yticks)
   ax1.tick_params(axis='y', rotation=90.0)
   ax1.set_xlabel(r'$l$')
   ax1.set_ylabel(r'$b$')
   plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)
   plt.savefig(filename+suffix+"_cartmap.png")
   plt.close()

   glonvec0=hdu.header['CDELT1']*(np.arange(hdu.header['NAXIS1'])-hdu.header['CRPIX1'])#+hdu.header['CRVAL1']
   glatvec0=hdu.header['CDELT2']*(np.arange(hdu.header['NAXIS2'])-hdu.header['CRPIX2'])#+hdu.header['CRVAL2']
   glonmat0, glatmat0 = np.meshgrid(glonvec,glatvec)

   fig = plt.figure(figsize=(13.0,6.5))
   plt.rc('font', size=10)
   ax1=plt.subplot(111, projection="mollweide")
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartImap, norm=colors.LogNorm(), cmap=planckct(), shading='auto')
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.2, shading='auto')
   ax1.set_xticks([0.], labels=np.round(l0,1))
   ax1.set_xlabel(r'$l$')
   ax1.set_ylabel(r'$b$')
   ax1.grid()
   plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)
   plt.savefig(filename+suffix+"mollmap.png")
   plt.close()
  
   fig = plt.figure(figsize=(13.0,6.5))
   plt.rc('font', size=10)
   ax1=plt.subplot(111, projection="mollweide")
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartImap, norm=colors.LogNorm(), cmap=planckct(), shading='auto')
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.2, shading='auto')
   ax1.set_xticks([]); ax1.set_yticks([]);
   plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
   plt.savefig(filename+suffix+"mollmapPR.png")
   plt.close()

   return cartImap, cartLICmap

# =============================================================================================
def lichppole(Imap, Qmap, Umap, l0=0., b0=0, reso=0.25, niter=3, slen=0.1, planck=True, filename='test'):

   # WARNING: DO NOT USE!!!
   # Aun no apto para consumo humano"

   deltal=135.0
   deltab=135.0
   dummy=np.zeros((int(deltab/reso),int(deltal/reso)))
   hdu=fits.PrimaryHDU(dummy)
   hdu.header['CDELT1']=-reso
   hdu.header['CTYPE1']='GLON-SIN'
   hdu.header['CRVAL1']=l0
   hdu.header['CRPIX1']=0.5*hdu.header['NAXIS1']
   hdu.header['CDELT2']=reso
   hdu.header['CTYPE2']='GLAT-SIN'
   hdu.header['CRVAL2']=-90.
   hdu.header['CRPIX2']=0.5*hdu.header['NAXIS2']

   cartImap, footprint = reproject_from_healpix((Imap, 'galactic'), hdu.header, nested=False) 
 
   plt.imshow(cartImap, norm=colors.LogNorm())
   plt.show()

   glonvec=hdu.header['CDELT1']*(np.arange(hdu.header['NAXIS1'])-hdu.header['CRPIX1'])+hdu.header['CRVAL1']
   glatvec=hdu.header['CDELT2']*(np.arange(hdu.header['NAXIS2'])-hdu.header['CRPIX2'])+hdu.header['CRVAL2']
   extent=np.array([np.max(glonvec),np.min(glonvec),np.min(glatvec),np.max(glatvec)])

   glonmat, glatmat = np.meshgrid(glonvec,glatvec)

   cartImap, footprint = reproject_from_healpix((Imap, 'galactic'), hdu.header, nested=False)
   cartQmap, footprint = reproject_from_healpix((Qmap, 'galactic'), hdu.header, nested=False)
   cartUmap, footprint = reproject_from_healpix((Umap, 'galactic'), hdu.header, nested=False)

   return cartImap, cartLICmap
  

