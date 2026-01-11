# This file is part of AstroHOG
#
# Copyright (C) 2025 Juan Diego Soler

import sys
import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy import constants as const

import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects

from scipy import ndimage
from scipy import signal
from scipy.stats import ks_2samp

import os
import imageio

import healpy as hp
import pycircstat

sys.path.append("/Users/soler/Documents/PYTHON/magnetar/")
from bvisual import *
from hppolkit import *
from healpix_lic import healpix_lic

# -------------------------------------------------------------------------------------
CHI=1.8224e18*u.cm**(-2)*(u.km/u.s)**(-1)
AVtoNH=5.8e21 # cm-2; Bohlin, Savage, Drake, et al. 

EtoAV=2.8*u.mag
AVtoNH=5.8e21*(1/u.cm**2)*(1/u.mag)
RV=3.1
dust2gas=1./100.0

NHtotau353=1.2e-26

# -------------------------------------------------------------------------------------
nside=2048
ipix=np.arange(hp.nside2npix(nside))
lvec, bvec=hp.pixelfunc.pix2ang(nside, ipix, lonlat=True)
hbpix=(np.abs(bvec) > 5.0).nonzero()
lbpix=(np.abs(bvec) <= 5.0).nonzero()

mppix=(np.abs(bvec) < 5.0).nonzero()
northpix=(bvec > 5.0).nonzero()
southpix=(bvec < -5.0).nonzero()

indir=''
dir2='/Users/soler/Documents/PYTHON/Planck/ForegroundMaps/Dust/'
hdup=fits.open(dir2+'COM_CompMap_IQU-thermaldust-gnilc-unires_2048_R3.00.fits')
I353=hdup[1].data['I_STOKES']*287.5 #-452
Q353=hdup[1].data['Q_STOKES']*287.5
U353=hdup[1].data['U_STOKES']*287.5
hdup.close()

hdup=fits.open(dir2+'COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits')
tau353=hdup[1].data['TAU353']
planckNH=tau353/NHtotau353

outhistPlanck=diagnosticHists(I353, Q353, U353, planckNH, polconv='Planck', label='Planck')

snpsht="824"
indir="/Users/soler/Documents/PYTHON/TressSims/example824/dust_Planck_loc2/data/"
inputname="TressSims"+snpsht+"_loc2"
hdu=fits.open(indir+"polaris_detector_nr0003.fits.gz")

Isim=hdu[1].data['I_STOKES (WAVELENGTH = 8.500000e-04 [m])']
Qsim=hdu[1].data['Q_STOKES (WAVELENGTH = 8.500000e-04 [m])']
Usim=hdu[1].data['U_STOKES (WAVELENGTH = 8.500000e-04 [m])']
NHsim=hdu[1].data['COLUMN_DENSITY']/1e2

soutput=smoothmaps(Isim, Qsim, Usim, 80., NHmap=NHsim)
outhistSim=diagnosticHists(soutput['Imap'], soutput['Qmap'], soutput['Umap'], soutput['NHmap'], polconv='Polaris', label='RheaIIsnapshot'+snpsht+"loc2")

prefix="PlanckAndRheaIIexample"

fig = plt.figure(figsize=(6.0,4.0))
plt.rc('font', size=14)
ax1=plt.subplot(111)
ax1.plot(outhistPlanck['binspsi'], outhistPlanck['histpsi'], color='dodgerblue', linewidth=2.0, label='Planck')
ax1.plot(outhistSim['binspsi'],    outhistSim['histpsi'], color='orange', linewidth=2.0, label=snpsht+" loc2")
ax1.axvline(x=0., linestyle='dashed')
ax1.tick_params(axis='y', labelrotation=90)
ax1.set_xlabel(r"$\psi$ [rad]")
ax1.set_ylabel(r"Counts")
plt.legend()
plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
plt.savefig(prefix+"_histPoverI.png")
plt.close()

fig = plt.figure(figsize=(6.0,4.0))
plt.rc('font', size=14)
ax1=plt.subplot(111)
ax1.plot(outhistPlanck['binsPoverI'], outhistPlanck['histPoverI'], color='dodgerblue', linewidth=2.0, label='Planck')
ax1.plot(outhistSim['binsPoverI'],    outhistSim['histPoverI'], color='orange', linewidth=2.0, label=snpsht+" loc2")
ax1.axvline(x=0., linestyle='dashed')
ax1.tick_params(axis='y', labelrotation=90)
ax1.set_xlabel(r"$P/I$ [%]")
ax1.set_ylabel(r"Counts")
plt.legend()
plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
plt.savefig(prefix+"_histPoverI.png")
plt.close()

fig = plt.figure(figsize=(6.0,4.0))
plt.rc('font', size=14)
ax1=plt.subplot(111)
ax1.plot(outhistPlanck['binsgradpsi'], outhistPlanck['histgradpsi'], color='dodgerblue', linewidth=2.0, label='Planck')
ax1.plot(outhistSim['binsgradpsi'],    outhistSim['histgradpsi'], color='orange', linewidth=2.0, label=snpsht+" loc2")
ax1.axvline(x=0., linestyle='dashed')
ax1.tick_params(axis='y', labelrotation=90)
ax1.set_xlabel(r"$\mathcal{S}$ [deg]")
ax1.set_ylabel(r"Counts")
plt.legend()
plt.subplots_adjust(left=0.1, bottom=0.14, right=0.99, top=0.94)
plt.savefig(prefix+"_histS.png")
plt.close()

import pdb; pdb.set_trace()


