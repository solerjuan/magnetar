# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2025 Juan Diego Soler

import numpy as np

from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.stats import circstats
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors

import healpy as hp
from tqdm import tqdm

from hptools import gradPsi

# -----------------------------------------------------------------
def map_angle_pi_to_halfpi(angle):
    """
    Maps angles from [-π, π] into [-π/2, π/2] by wrapping.
    Accepts scalars or NumPy arrays.
    """
    angle = np.asarray(angle)

    # Normalize to [-π, π]
    angle_norm = (angle + np.pi) % (2 * np.pi)
    angle_norm = np.where(angle_norm < 0, angle_norm + 2 * np.pi, angle_norm)
    angle_norm = angle_norm - np.pi

    half_pi = np.pi / 2

    # Wrap values outside [-π/2, π/2]
    wrapped = np.where(angle_norm > half_pi,
                       angle_norm - np.pi,
                       np.where(angle_norm < -half_pi,
                                angle_norm + np.pi,
                                angle_norm))
    return wrapped

# -----------------------------------------------------------------
def rotate_and_wrap_90(angle):
    """
    Rotates an angle (in radians) by +90 degrees (π/2)
    and wraps it into the range [-π/2, π/2].
    Works for scalars or NumPy arrays.
    """
    angle = np.asarray(angle)

    # Rotate by +90 degrees (π/2 radians)
    rotated = angle + np.pi / 2

    # Normalize to [-π, π]
    norm = (rotated + np.pi) % (2 * np.pi)
    norm = np.where(norm < 0, norm + 2 * np.pi, norm)
    norm = norm - np.pi

    # Wrap into [-π/2, π/2]
    half_pi = np.pi / 2
    wrapped = np.where(norm > half_pi,
                       norm - np.pi,
                       np.where(norm < -half_pi,
                                norm + np.pi,
                                norm))
    return wrapped

# ===================================================================================================
def smoothmaps(Imap, Qmap, Umap, fwhm, fwhm0=0., NHmap=None):

   """
    Smooth Healpix Stokes parameter maps

    Parameters
    ----------
    Imap, Qmap, Umap : 
        Height and width of the output array.
    fwhm : float
        Selected beam size in arminutes
    fwhm0: float
        Initial beam size in arcminutes (=0. for sims).

    Returns
    -------
    window : ndarray of shape (h, w)
        Binary array with 1s inside the rectangle and 0s elsewhere.
   """
  
   if (fwhm < fwhm0):
      print("Selected beam size must be greater than the initial beam size") 
      return None

   fwhmI=np.sqrt(fwhm**2-fwhm0**2)

   sImap=hp.sphtfunc.smoothing(Imap.copy(), fwhm=np.deg2rad(fwhmI/60.))
   sQmap=hp.sphtfunc.smoothing(Qmap.copy(), fwhm=np.deg2rad(fwhmI/60.))
   sUmap=hp.sphtfunc.smoothing(Umap.copy(), fwhm=np.deg2rad(fwhmI/60.))

   if (NHmap is None):
      sNHmap=None
   else:
      sNHmap=hp.sphtfunc.smoothing(NHmap.copy(), fwhm=np.deg2rad(fwhmI/60.)) 

   return {'Imap': sImap, 'Qmap': sQmap, 'Umap': sUmap, 'NHmap': sNHmap}

# ===================================================================================================
def diagnosticHists(Imap, Qmap, Umap, NHmap, polconv='Polaris', label='Test', niter=3): 

   if (polconv=='Polaris'):
      psimap=rotate_and_wrap_90(0.5*np.arctan2(Umap,Qmap))
   if (polconv=='Planck'):
      psimap=map_angle_pi_to_halfpi(0.5*np.arctan2(Umap,Qmap)) 

   histpsi, bins = np.histogram(psimap, range=[-np.pi/2.,np.pi/2.], bins=90)
   binspsi=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)]) 
  
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.plot(binspsi, histpsi, color='blue', linewidth=2.0, label=label)
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\psi$ [rad]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histPsi.png")
   plt.close()

   PoverImap=np.sqrt(Qmap**2+Umap**2)/Imap
   histPoverI, bins = np.histogram(100*PoverImap, range=[0,25.0], bins=100)  
   binsPoverI=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)]) 

   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.plot(binsPoverI, histPoverI, color='orange', linewidth=2.0, label=label)
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$P/I$ [%]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histPoverI.png")
   plt.close()
    
   binspoveri=np.linspace(0.,30.,500)
   binslognh=np.linspace(-2.0,2.0,500)

   NHmap[(NHmap < 1.).nonzero()]=np.nan 
   logNH21=np.log10(NHmap/1e21)
   good=np.logical_and(np.isfinite(logNH21),np.isfinite(PoverImap)).nonzero()
   hist2DlognhANDpoveri, xedges, yedges = np.histogram2d(logNH21[good], 100.*PoverImap[good], bins=(binslognh,binspoveri))

   xmat, ymat = np.meshgrid(xedges, yedges)

   fig = plt.figure(figsize=(6.0,5.5))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   im1=ax1.pcolormesh(xmat, ymat, hist2DlognhANDpoveri.T, norm=colors.LogNorm(), cmap='jet')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\log_{10}(N_{\rm H}/10^{21}$ cm$^{-2}$)")
   ax1.set_ylabel(r"$P/I$ [%]")
   ax_cb=ax1.inset_axes([1.025, 0.0, 0.03, 1.0])
   cbar=plt.colorbar(im1, ax=ax1, cax=ax_cb)
   cbar.ax.tick_params(axis='y', labelrotation=90)
   cbar.ax.set_title(r'$T_{\rm b}$ [K]', fontsize=12, ha='center')
   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.96)
   plt.tight_layout()
   plt.savefig(label+"_hist2DlognhANDpoveri.png")
   plt.close()

   print("Calculating NablaPsi")
   ksz=1.0 #deg
   gradpsi=gradPsi(Qmap, Umap, ksz=ksz, niter=niter)
   Sfunc=np.rad2deg(np.deg2rad(ksz)*gradpsi/(2.*np.sqrt(2.)))
 
   histgradpsi, bins = np.histogram(Sfunc, bins=100) 
   binsgradpsi=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)])
   
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.plot(binsgradpsi, histgradpsi, color='magenta', linewidth=2.0, label=label)
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\mathcal{S}$ [deg]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histS.png")
   plt.close()
   #import pdb; pdb.set_trace()

   return {'binspsi': binspsi, 'histpsi': histpsi, 'binPoverI': binPoverI, 'histPoverI': histPoverI, 'binsgradpsi': binsgradpsi, 'histgradpsi': histgradpsi}

# ===================================================================================================
#def diagnostic2DHists(Imap, Qmap, Umap, fwhm, fwhm0=0.):

  
