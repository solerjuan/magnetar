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
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import EllipticalFrame

from congrid import *

import healpy as hp
from healpy.newvisufunc import projview
from reproject import reproject_from_healpix, reproject_to_healpix

from scipy import interpolate
from tqdm import tqdm

sys.path.append('/Users/soler/Documents/PYTHON/magnetar/')
from bvisual import planckct, lic

def wrap_to_360(angle_deg):
    return angle_deg % 360

def wrap_to_360_inclusive(angle_deg):
    wrapped = angle_deg % 360
    wrapped[wrapped == 0] = 360
    return wrapped

def wrap_to_180(angle_deg):
    return (angle_deg + 180) % 360 - 180

def wrap_to_90(angle_deg):
    return ((angle_deg + 90) % 180) - 90

# =============================================================================================
def lichpMoll(Imap, Qmap, Umap, l0=0., b0=0., reso=0.25, niter=3, slen=0.1, planck=True, filename='test', vmin=None, vmax=None):

   if (vmin is None):
      vmin=np.nanmin(Imap)

   if (vmax is None):
      vmax=np.nanmax(Imap)

   suffix='_l'+str(l0)+'b'+str(b0)+'_LICreso'+str(reso)+'niter'+str(niter)+'slen'+str(slen)

   #deltal=360.0
   #deltab=180.0
   #dummy=np.zeros((int(deltab/reso),int(deltal/reso)))
   hdu=fits.PrimaryHDU()
   hdu.header['NAXIS']=2
   hdu.header['NAXIS1']=800
   hdu.header['NAXIS2']=400
   hdu.header['CTYPE1']='GLON-MOL'
   hdu.header['CRPIX1']=hdu.header['NAXIS1']/2
   hdu.header['CRVAL1']=l0
   hdu.header['CDELT1']=-360./hdu.header['NAXIS1']
   hdu.header['CUNIT1']='deg     '
   hdu.header['CTYPE2']='GLAT-MOL'
   hdu.header['CRPIX2']=hdu.header['NAXIS2']/2
   hdu.header['CRVAL2']=b0
   hdu.header['CDELT2']=180./hdu.header['NAXIS2']
   hdu.header['CUNIT2']='deg     '
   hdu.header['COORDSYS']='Galactic'
   target_header=hdu.header.copy()

   hp.fitsfunc.write_map('output.fits', Imap, coord='G', overwrite=True)
   mollImap, footprint = reproject_from_healpix('output.fits', target_header)
   import pdb; pdb.set_trace()
   
   fig = plt.figure(figsize=(12.0, 6.0))
   plt.rc('font', size=10)
   ax=plt.subplot(111, projection=WCS(target_header), frame_class=EllipticalFrame)
   ax.imshow(mollImap, interpolation='none', cmap=planckct(), norm=colors.LogNorm(), zorder=0)

# =============================================================================================
def lichp(Imap, Qmap, Umap, l0=0., b0=0., reso=0.25, niter=3, slen=0.1, planck=True, filename='test', vmin=None, vmax=None):

   if (vmin is None):
      vmin=np.nanmin(Imap)

   if (vmax is None):
      vmax=np.nanmax(Imap)

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

   glonvec=wrap_to_360_inclusive(hdu.header['CDELT1']*(np.arange(hdu.header['NAXIS1'])-hdu.header['CRPIX1'])+hdu.header['CRVAL1'])
   glatvec=wrap_to_90(hdu.header['CDELT2']*(np.arange(hdu.header['NAXIS2'])-hdu.header['CRPIX2'])+hdu.header['CRVAL2'])
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

   glonvec0=hdu.header['CDELT1']*(np.arange(hdu.header['NAXIS1'])-hdu.header['CRPIX1'])#+hdu.header['CRVAL1']
   glatvec0=hdu.header['CDELT2']*(np.arange(hdu.header['NAXIS2'])-hdu.header['CRPIX2'])#+hdu.header['CRVAL2']
   glonmat0, glatmat0 = np.meshgrid(glonvec0,glatvec0)

   fig = plt.figure(figsize=(13.0,6.5), dpi=300)
   plt.rc('font', size=10)
   ax1=plt.subplot(111)
   ax1.pcolormesh(np.deg2rad(wrap_to_180(-glonmat0)), np.deg2rad(wrap_to_90(glatmat0)), cartImap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=planckct(), shading='nearest')
   ax1.pcolormesh(np.deg2rad(wrap_to_180(-glonmat0)), np.deg2rad(wrap_to_90(glatmat0)), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.25, shading='nearest')
   xticks=np.deg2rad(wrap_to_180(np.arange(-180.1,180.1,90)))
   ax1.set_xticks(xticks, labels=np.round(wrap_to_360(np.arange(-180.1+l0,180.1+l0,90.))[::-1]))
   yticks=np.deg2rad(wrap_to_90(np.arange(-90.,90.1,45)))
   ax1.set_yticks(yticks, labels=np.round(wrap_to_90(np.arange(-90.1+b0,90.1+b0,45.))))
   ax1.tick_params(axis='y', rotation=90.0)
   ax1.set_xlabel(r'$l$')
   ax1.set_ylabel(r'$b$')
   plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)
   plt.savefig(filename+suffix+"_cartmap.png")
   plt.close()
   
   # ---------------------------------------------------------
   fig = plt.figure(figsize=(13.0,6.5), dpi=300)
   plt.rc('font', size=10)
   ax1=plt.subplot(111, projection="mollweide")
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartImap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=planckct(), shading='auto')
   ax1.pcolormesh(-np.deg2rad(glonmat0), np.deg2rad(glatmat0), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.25, shading='auto')
   xticks=np.deg2rad(wrap_to_180(np.arange(-180.1,180.1,90)))
   ax1.set_xticks(xticks, labels=np.round(wrap_to_360(np.arange(-180.1+l0,180.1+l0,90.))[::-1]))
   yticks=np.deg2rad(wrap_to_90(np.arange(-90.,90.1,30.)))
   ax1.set_yticks(yticks, labels=np.round(wrap_to_90(np.arange(-90.1+b0,90.1+b0,30.))))
   ax1.set_xlabel(r'$l$')
   ax1.set_ylabel(r'$b$')
   ax1.grid()
   plt.subplots_adjust(left=0.08, bottom=0.08, right=0.97, top=0.97)
   plt.savefig(filename+suffix+"mollmap.png")
   plt.close()
  
   fig = plt.figure(figsize=(13.0,6.5), dpi=300)
   plt.rc('font', size=10)
   ax1=plt.subplot(111, projection="mollweide")
   ax1.pcolormesh(np.deg2rad(wrap_to_180(-glonmat0)), np.deg2rad(glatmat0), cartImap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=planckct(), shading='auto')
   ax1.pcolormesh(np.deg2rad(wrap_to_180(-glonmat0)), np.deg2rad(glatmat0), cartLICmap, vmin=licmin, vmax=licmax, cmap='binary', alpha=0.25, shading='auto')
   ax1.set_xticks([]); ax1.set_yticks([]);
   plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
   plt.savefig(filename+suffix+"mollmapPR.png")
   plt.close()
   import pdb; pdb.set_trace()
   return {'Imap': cartImap, 'LICmap': cartLICmap, 'glonmat': glonmat, 'glatmat': glatmat}

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
  
# -----------------------------
# Helper: spherical integration
# -----------------------------
def advance_on_sphere(theta, phi, v_theta, v_phi, ds):
    """
    Advance a position (theta,phi) on the sphere by a small angular step ds
    following the tangential vector components (v_theta, v_phi).
    - theta: colatitude [0..pi]
    - phi: longitude [0..2pi)
    - v_theta, v_phi: tangential vector components (units: 1, treated as direction)
    - ds: step length in radians (controls arc length)
    Returns new (theta, phi) normalized into ranges.
    """
    # small-angle update: dtheta = v_theta * ds
    # dphi = v_phi * ds / sin(theta)  (to convert tangential phi-component to angular change)
    # guard sin(theta)=0 (poles)
    st = np.sin(theta)
    dtheta = v_theta * ds
    # avoid division by zero at poles: if sin(theta) ~ 0, set dphi = v_phi*ds/(small)
    eps = 1e-8
    dphi = v_phi * ds / (st + eps)

    theta_new = theta + dtheta
    phi_new = phi + dphi

    # clamp theta into [0, pi], reflect at poles to keep continuity
    # If theta crosses a pole, flip theta and add pi to phi
    if theta_new < 0:
        theta_new = -theta_new
        phi_new += np.pi
    elif theta_new > np.pi:
        theta_new = 2*np.pi - theta_new
        phi_new += np.pi

    # normalize phi into [0, 2pi)
    phi_new = np.mod(phi_new, 2*np.pi)

    return theta_new, phi_new

# -----------------------------
# LIC function
# -----------------------------
def healpix_lic(v_theta_map, v_phi_map, nside,
                noise_map=None,
                length=0.2,    # integration length in radians (total forward+backwards half-length)
                ds=0.005,      # step size in radians
                kernel='box'   # 'box' or 'gaussian'
               ):
    """
    Compute LIC on a HEALPix grid.
    v_theta_map, v_phi_map : arrays length Npix containing tangential components
       (components in theta, phi directions). Units are arbitrary; only direction matters.
    nside : healpix nside for maps.
    noise_map : optional scalar map as texture. If None, white noise is generated.
    length : total convolution half-length (radians). The LIC samples forward and backward length each.
    ds : step increment along streamline (radians).
    kernel : integration kernel type. 'box' or 'gaussian'.

    Returns: lic_map (float map)
    """

    npix = hp.nside2npix(nside)

    # Create noise if not provided
    if noise_map is None:
        rng = np.random.default_rng(0)
        noise_map = rng.standard_normal(npix)

    # Precompute pixel angles
    thetas, phis = hp.pix2ang(nside, np.arange(npix))  # theta: colatitude, phi: lon

    # For interpolation of noise and vector components we will use healpy.get_interp_val
    # which takes arrays (map, theta, phi) and returns interpolated values.
    # However healpy.get_interp_val expects theta, phi as scalars or arrays; we will call it for each sample.

    # Precompute sampling kernel weights
    n_steps = int(np.ceil(length / ds))
    # total number of samples along streamline = 2*n_steps + 1 (backwards, centre, forward)
    if kernel == 'box':
        k_weights = np.ones(2*n_steps + 1)
    elif kernel == 'gaussian':
        # gaussian over arc-length
        xs = np.linspace(-n_steps*ds, n_steps*ds, 2*n_steps + 1)
        sigma = length/2.0 if length>0 else ds
        k_weights = np.exp(-0.5 * (xs/sigma)**2)
    else:
        raise ValueError("Unsupported kernel: choose 'box' or 'gaussian'")

    # normalize kernel
    k_weights = k_weights / np.sum(k_weights)

    lic = np.zeros(npix, dtype=float)

    # main loop over pixels
    for pix in tqdm(range(npix)):
        theta0 = thetas[pix]
        phi0 = phis[pix]

        # center sample
        acc = 0.0
        wsum = 0.0

        # run backward and forward
        # we'll start from center and step +/- ds along streamlines
        # First sample center
        noise_center = hp.get_interp_val(noise_map, theta0, phi0)
        acc += k_weights[n_steps] * noise_center
        wsum += k_weights[n_steps]

        # Backward direction: reverse vector
        theta = theta0
        phi = phi0
        for s in range(1, n_steps+1):
            # get local vector at current position
            # Interpolate v_theta and v_phi
            vth = hp.get_interp_val(v_theta_map, theta, phi)
            vph = hp.get_interp_val(v_phi_map, theta, phi)

            # step in negative direction
            theta, phi = advance_on_sphere(theta, phi, -vth, -vph, ds)
            sample = hp.get_interp_val(noise_map, theta, phi)
            weight = k_weights[n_steps - s]
            acc += weight * sample
            wsum += weight

        # Forward direction
        theta = theta0
        phi = phi0
        for s in range(1, n_steps+1):
            vth = hp.get_interp_val(v_theta_map, theta, phi)
            vph = hp.get_interp_val(v_phi_map, theta, phi)

            theta, phi = advance_on_sphere(theta, phi, vth, vph, ds)
            sample = hp.get_interp_val(noise_map, theta, phi)
            weight = k_weights[n_steps + s]
            acc += weight * sample
            wsum += weight

        lic[pix] = acc / (wsum + 1e-15)

    # Normalize result to 0..1 for display
    lic -= lic.min()
    if lic.max() > 0:
        lic /= lic.max()
    return lic





