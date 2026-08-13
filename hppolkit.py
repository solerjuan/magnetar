# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2025 Juan Diego Soler

import sys
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

sys.path.append("../")
from bvisual import *

# ----------------------------------------------------------------
def angle_difference(angle1, angle2):

    diff = np.deg2rad(angle2 - angle1)
    return np.rad2deg(np.arctan2(np.sin(diff), np.cos(diff)))

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

   print("Smoothing I map")
   sImap=hp.sphtfunc.smoothing(Imap.copy(), fwhm=np.deg2rad(fwhmI/60.))
   print("Smoothing Q map")
   sQmap=hp.sphtfunc.smoothing(Qmap.copy(), fwhm=np.deg2rad(fwhmI/60.))
   print("Smoothing U map")
   sUmap=hp.sphtfunc.smoothing(Umap.copy(), fwhm=np.deg2rad(fwhmI/60.))

   if (NHmap is None):
      sNHmap=None
   else:
      sNHmap=hp.sphtfunc.smoothing(NHmap.copy(), fwhm=np.deg2rad(fwhmI/60.)) 

   return {'Imap': sImap, 'Qmap': sQmap, 'Umap': sUmap, 'NHmap': sNHmap}

# ===================================================================================================
def diagnosticHists(Imap, Qmap, Umap, NHmap, polconv='Polaris', label='Test', niter=3, nglatbins=5, nest=False): 

   if (polconv=='Polaris'):
      psimap=rotate_and_wrap_90(0.5*np.arctan2(Umap,Qmap))
   if (polconv=='Planck'):
      psimap=map_angle_pi_to_halfpi(0.5*np.arctan2(Umap,Qmap)) 

   ipix=np.arange(np.size(Imap))
   glon, glat= hp.pix2ang(hp.npix2nside(np.size(Imap)), ipix, lonlat=True)

   scaleheight=150.0
   distance=500.0
   refang=np.arctan(0.5*scaleheight/distance)      

   # ===============================================================================
   PoverImap=np.sqrt(Qmap**2+Umap**2)/Imap
   histPoverI, bins = np.histogram(100*PoverImap, range=[0,35.0], bins=500, density=True)
   binsPoverI=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)])

   # P/I map (Fig. 2. in Planck 2018 XI)
   hp.mollview(100*PoverImap, min=0., max=25., unit=r'$P/I$ [%]', cmap=planckct(), nest=nest) 
   hp.graticule()
   plt.savefig(label+"_mapPoverI.png")
   plt.close()

   # P/I histogram (Fig. 5. in Planck 2018 XI)
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_xlim(0.,31.)
   ax1.set_ylim(2e-3,0.2)
   ax1.semilogy(binsPoverI, histPoverI, color='orange', linewidth=2.0, label=label)
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$P/I$ [%]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histPoverI.png")
   plt.close()
 
   hglat, bglat = np.histogram(glat, bins=1000) 
   bcglat=0.5*(bglat[0:np.size(bglat)-1]+bglat[1:np.size(bglat)])
   cumhglat=np.cumsum(hglat)/np.sum(hglat)

   glatbins=np.zeros(nglatbins+1)
   for i in range(0,nglatbins+1): diff=np.abs(cumhglat-i*(1/nglatbins)); ibin=np.min((diff==np.min(diff)).nonzero()); glatbins[i]=bcglat[ibin]

   # -----------------------------------------------------
   histsPoverI=np.zeros([nglatbins,np.size(bins)-1])   
   for i in range(0,nglatbins):
      temphist, bins = np.histogram(100*PoverImap[np.logical_and(glat > glatbins[i], glat < glatbins[i+1]).nonzero()], bins=bins, density=True) 
      histsPoverI[i,:]=temphist
 
   mycolors=plt.cm.managua(np.linspace(0, 1, nglatbins))

   # P/I histograms in glat ranges
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_xlim(0.,31.)
   ax1.set_ylim(2e-3,0.2)
   for i in range(0,nglatbins): ax1.semilogy(binsPoverI, histsPoverI[i,:], color=mycolors[i], linewidth=2.0, label=str(np.round(glatbins[i]))+r"$<b<$"+str(np.round(glatbins[i+1]))+r"$^{\circ}$")
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$P/I$ [%]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histPoverImultib.png")
   plt.close() 

   # Galactic plane glon bins 
   bmin=-10.0; bmax=10.0
   #bmin=-5.0; bmax=5.0
   #lbins=np.linspace(0.,360.0,6)
   deltal=45.
   lbcen=np.arange(0.,360.,deltal)

   histsPoverI=np.zeros([np.size(lbcen),np.size(bins)-1]) 

   for i in range(0,np.size(lbcen)):
      goodb=np.logical_and(glat > bmin, glat < bmax)
      diff=np.abs(angle_difference(glon, lbcen[i]))
      goodl=(diff < 0.5*deltal)
      temphist, bins = np.histogram(100*PoverImap[np.logical_and(goodb, goodl).nonzero()], bins=bins, density=True)
      histsPoverI[i,:]=temphist

      #mask=np.zeros_like(glat) 
      #mask[np.logical_and(goodb, goodl).nonzero()]=1.
      #hp.mollview(mask) 
      #plt.show()

   mycolors=plt.cm.hsv(np.linspace(0, 1, np.size(lbcen)+1))

   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_title(str(bmin)+r"$<b<$"+str(bmax)+r"$^{\circ}$")
   ax1.set_xlim(0.,31.)
   ax1.set_ylim(2e-3,np.nanmax(1.1*histsPoverI))
   for i in range(0,np.size(lbcen)): ax1.semilogy(binsPoverI, histsPoverI[i,:], color=mycolors[i], linewidth=2.0, label=str(np.round(lbcen[i]-0.5*deltal))+r"$<l<$"+str(np.round(lbcen[i]+0.5*deltal))+r"$^{\circ}$")
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$P/I$ [%]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.92)
   plt.savefig(label+"_histPoverIplanelbins.png")
   plt.close()

   # =============================================================================== 
   deltal=360./180
   lsteps=np.arange(-180.,180.,deltal)
   bsteps=np.zeros_like(lsteps)

   selpix=hp.ang2pix(hp.npix2nside(np.size(Imap)), lsteps, bsteps, lonlat=True)

   fig = plt.figure(figsize=(6.0,5.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(211)
   ax1.set_xlim(180.,-180.)
   ax1.plot(lsteps, Imap[selpix], color='orange')
   ax1.axvline(x=90., color='grey', linestyle='dashed', alpha=0.3)
   ax1.axvline(x=0., color='grey', linestyle='dashed', alpha=0.3)
   ax1.axvline(x=-90., color='grey', linestyle='dashed', alpha=0.3)
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_ylabel(r"$I$ [Jy/sr]")
   ax2=plt.subplot(212)
   ax2.set_xlim(180.,-180.)
   ax2.plot(lsteps, 100*np.sqrt(Qmap[selpix]**2+Umap[selpix]**2)/Imap[selpix], color='dodgerblue')
   ax2.axvline(x=90., color='grey', linestyle='dashed', alpha=0.3)
   ax2.axvline(x=0., color='grey', linestyle='dashed', alpha=0.3)
   ax2.axvline(x=-90., color='grey', linestyle='dashed', alpha=0.3)
   ax2.tick_params(axis='y', labelrotation=90)
   ax2.set_ylabel(r"$P/I$ [%]")
   ax2.set_xlabel(r"$l$ [deg]")
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.99)
   plt.savefig(label+"_planeIandPoverIprofiles.png")
   plt.close()

   # ===============================================================================
   histpsi, bins = np.histogram(psimap, range=[-np.pi/2.,np.pi/2.], bins=500, density=True)
   binspsi=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)]) 

   # psi map (Fig. 2. in Planck 2018 XI)
   hp.mollview(np.rad2deg(psimap), min=-90., max=90., unit=r'$\psi$ [deg]', cmap='hsv', nest=nest)
   hp.graticule()
   plt.savefig(label+"_mapPsi.png")
   plt.close()

   # Psi histogram (Fig. 6. in Planck 2018 XI)   
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_xlim(-90.,90.)
   ax1.set_ylim(0.0,1.0) 
   ax1.plot(np.rad2deg(binspsi), histpsi, color='blue', linewidth=2.0, label=label)
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\psi$ [rad]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histPsi.png")
   plt.close()

   # ===============================================================================
   print("Calculating NablaPsi")
   ksz=80.0/60.0 #deg
   gradpsi=gradPsi(Qmap, Umap, ksz=ksz, niter=niter)
   Sfunc=np.rad2deg(np.deg2rad(ksz)*gradpsi/(2.*np.sqrt(2.)))

   # psi map (Fig. 2. in Planck 2018 XI)
   hp.mollview(Sfunc, norm='log', unit=r'$\log_{10}(\mathcal{S}/{\rm deg})$', cmap='gist_heat', min=0.1, max=60., nest=nest)
   hp.graticule()
   plt.savefig(label+"_mapS.png")
   plt.close()
 
   histS, bins = np.histogram(Sfunc, bins=500, density=True) 
   binsS=0.5*(bins[0:np.size(bins)-1]+bins[1:np.size(bins)])
   
   # S histogram (Fig. 7. in Planck 2018 XI) 
   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_xlim(0.,90.) 
   ax1.set_ylim(8e-5,0.3)
   ax1.semilogy(binsS, histS, color='magenta', linewidth=2.0, label=label)
   ax1.axvline(x=52., color='grey', linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\mathcal{S}$ [deg]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histS.png")
   plt.close()

   bin1=(PoverImap <= 0.01).nonzero()
   bin2=np.logical_and(PoverImap >0.01, PoverImap < 0.05).nonzero()
   bin3=(PoverImap >= 0.05).nonzero()

   histS1, bins1 = np.histogram(Sfunc[bin1], bins=bins, density=True)
   histS2, bins2 = np.histogram(Sfunc[bin2], bins=bins, density=True)
   histS3, bins3 = np.histogram(Sfunc[bin3], bins=bins, density=True)

   fig = plt.figure(figsize=(6.0,4.0))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   ax1.set_xlim(0.,90.)  
   ax1.set_ylim(8e-5,0.3)
   ax1.semilogy(binsS, histS, color='black', linewidth=2.0, label='All')
   ax1.semilogy(binsS, histS1, color='red', linewidth=2.0, label=r'$p < 1\%$')
   ax1.semilogy(binsS, histS2, color='blue', linewidth=2.0, label=r'$1 < p < 5\%$')
   ax1.semilogy(binsS, histS3, color='teal', linewidth=2.0, label=r'$p > 5\%$')  
   ax1.axvline(x=0., linestyle='dashed')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\mathcal{S}$ [deg]")
   ax1.set_ylabel(r"Counts")
   plt.legend()
   plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
   plt.savefig(label+"_histSmultiPoverI.png")
   plt.close()

   # ===============================================================================
   binsp=np.linspace(0.,30.,50)
   binslognh=np.linspace(-2.0,2.0,50)

   NHmap[(NHmap < 1.).nonzero()]=np.nan 
   logNH21=np.log10(NHmap/1e21)
   good=np.logical_and(np.isfinite(logNH21),np.isfinite(PoverImap)).nonzero()
   hist2DlognhANDpoveri, xedges, yedges = np.histogram2d(logNH21[good], 100.*PoverImap[good], bins=(binslognh,binsp))

   xmat, ymat = np.meshgrid(xedges, yedges)

   # P/I and NH 2D histogram (Fig. 9. in Planck 2018 XI) 
   fig = plt.figure(figsize=(6.0,5.5))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   #im1=ax1.pcolormesh(xmat, ymat, hist2DlognhANDpoveri.T, norm=colors.LogNorm(), cmap='jet')
   im1=ax1.pcolormesh(xmat, ymat, np.log10(hist2DlognhANDpoveri).T, cmap='jet')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$\log_{10}(N_{\rm H}/10^{21}$ cm$^{-2}$)")
   ax1.set_ylabel(r"$P/I$ [%]")
   ax_cb=ax1.inset_axes([1.025, 0.0, 0.03, 1.0])
   cbar=plt.colorbar(im1, ax=ax1, cax=ax_cb)
   cbar.ax.tick_params(axis='y', labelrotation=90)
   cbar.ax.set_title(r'log(counts)', fontsize=12, ha='center')
   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.96)
   plt.tight_layout()
   plt.savefig(label+"_hist2DlognhANDpoveri.png")
   plt.close()
   
   # ===============================================================================
   pmin=0.1; pmax=25.0
   Smin=0.2; Smax=90.0

   binsp=np.linspace(np.log10(pmin),np.log10(pmax),100)
   binsS2=np.linspace(np.log10(Smin),np.log10(Smax),100)
 
   logPoverI=np.log10(100*PoverImap)
   logS=np.log10(Sfunc)

   good=np.logical_and(np.isfinite(logPoverI),np.isfinite(logS)).nonzero()
   hist2DpoveriANDs, xedges, yedges = np.histogram2d(logPoverI[good], logS[good], bins=(binsp,binsS2))

   xmat, ymat = np.meshgrid(xedges, yedges)

   hist2DpoveriANDs[(hist2DpoveriANDs < 1.0).nonzero()]=np.nan

   # P/I and NH 2D histogram (Fig. 10. in Planck 2018 XI) 
   fig = plt.figure(figsize=(6.0,5.5))
   plt.rc('font', size=14)
   ax1=plt.subplot(111)
   im1=ax1.pcolormesh(xmat, ymat, np.log10(hist2DpoveriANDs.T), cmap='jet')
   ax1.tick_params(axis='y', labelrotation=90)
   ax1.set_xlabel(r"$P/I$ [%]")
   ax1.set_ylabel(r"$\mathcal{S}$ [deg]")  
   ax_cb=ax1.inset_axes([1.025, 0.0, 0.03, 1.0])
   cbar=plt.colorbar(im1, ax=ax1, cax=ax_cb)
   cbar.ax.tick_params(axis='y', labelrotation=90)
   cbar.ax.set_title(r'log(counts)', fontsize=12, ha='center')
   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.96)
   plt.tight_layout()
   plt.savefig(label+"_hist2DpoveriANDs.png")
   plt.close()

   #import pdb; pdb.set_trace()

   return {'binspsi': binspsi, 'histpsi': histpsi, 'binsPoverI': binsPoverI, 'histPoverI': histPoverI, 'binsgradpsi': binsS, 'histgradpsi': histS}

# ===================================================================================================
#def diagnostic2DHists(Imap, Qmap, Umap, fwhm, fwhm0=0.):

  
