# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.stats import circstats
from scipy import ndimage

import pycircstat as circ
from tqdm import tqdm

from bvisual import *
from poltools import *

# ===================================================================================================
def roangles(Imap, Qmap, Umap, ksz=1, mask=0, mode='reflect', convention='Planck', debug=False):
    # Calculates the relative orientation angle between the density structures and the magnetic field following the method
    # presented in Soler, et al. ApJ 774 (2013) 128S
    #
    # INPUTS
    # Imap - Intensity or column density map
    # Qmap - Stokes Q map
    # Upam - Stokes U map
    # OUTPUTS
    # phi - relative orientation angle between the column density and projected magnetic field.

    if (mask is 0):
       mask=np.ones_like(Imap)
   
    psi=0.5*np.arctan2(Umap,Qmap)	
    ex=-np.sin(psi)
    ey=np.cos(psi) 
    angleb=np.arctan2(ey,-ex)
    bx=ey; by=-ex

    sImap=ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[0,0], mode=mode)
    dIdx =ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[0,1], mode=mode)
    dIdy =ndimage.filters.gaussian_filter(Imap, [ksz, ksz], order=[1,0], mode=mode)
    angleg=np.arctan2(dIdx,dIdy)

    normgrad=np.sqrt(dIdx**2+dIdy**2)
    unidIdx=dIdx/np.sqrt(dIdx**2+dIdy**2)
    unidIdy=dIdy/np.sqrt(dIdx**2+dIdy**2)

    cosphi=(dIdy*ey+dIdx*ex)/normgrad
    sinphi=(dIdy*ex-dIdx*ey)/normgrad

    #phi=np.arctan2(np.abs(dIdy*ex-dIdx*ey), dIdy*ey+dIdx*ex)
    phi=np.arctan2(np.abs(sinphi), cosphi)
    #phi=np.arccos(cosphi)
    #phi=np.arcsin(sinphi)
    bad=((dIdx**2+dIdy**2)==0.).nonzero()    #np.logical_or((dIdx**2+dIdy**2)==0., (Qmap**2+Umap**2)==0.).nonzero()	
    phi[bad]=np.nan
    bad=((Qmap**2+Umap**2)==0.).nonzero()
    phi[bad]=np.nan
    bad=(mask < 1.).nonzero()
    phi[bad]=np.nan

    # ------------------------------------------------------------------------------------
    vecpitch=10
    xx, yy, gx, gy = vectors(Imap, dIdx*mask, dIdy*mask, pitch=vecpitch) 
    xx, yy, ux, uy = vectors(Imap, bx*mask, by*mask, pitch=vecpitch)   
    if np.array_equal(np.shape(Imap), np.shape(mask)):
       bad=(mask==0.).nonzero()
       phi[bad]=np.nan    
 
    # Debugging -------------------------------------------------------------------------
    if (debug):
       levels=np.nanmean(Imap)+np.array([0.,1.,2.,3.,5.,7.])*np.nanstd(Imap)

       fig = plt.figure(figsize=(7.0,7.0))
       plt.rc('font', size=10)
       ax1=plt.subplot(111)
       im=ax1.imshow(np.abs(np.rad2deg(np.arctan(np.tan(phi)))), origin='lower', cmap='cividis')
       ax1.quiver(xx, yy, gx, gy, units='width', color='red',   pivot='middle', scale=25., headlength=0, headwidth=1, label=r'$\nabla I$')
       ax1.quiver(xx, yy, ux, uy, units='width', color='black', pivot='middle', scale=25., headlength=0, headwidth=1, label=r'$B_{\perp}$')
       ax1.contour(Imap, origin='lower', colors='black', levels=levels, linewidths=1.0)
       cbar=fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
       cbar.ax.set_title(r'$\phi$')
       plt.legend()
       plt.show()
       import pdb; pdb.set_trace()

    return np.arctan(np.tan(phi))

# ===================================================================================================
def roparameterhist(phi, hsize=15, s_phi=20.):
    # Calculate the relative orientation parameter $\xi$ as defined in Planck intermediate results. XXXV. A&A 586A (2016) 138P.
    #
    # INPUTS
    # phi     - vector with the reference values for the histogram
    # s_phi   - range for the definitions of parallel (0 < phi < s_phi and 180-s_phi < phi < 180) or 
    #           perpendicular (90-s_phi < phi < 90+s_phi)
    # OUTPUTS
    # xi 	  - relative orientation parameter

    hist, bin_edges = np.histogram((180/np.pi)*phi, bins=hsize, range=(-90.,90.))
    bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

    perp=(np.abs(bin_centre) > 90.-s_phi).nonzero()
    para=(np.abs(bin_centre) < s_phi).nonzero()
    xi=float(np.mean(hist[para])-np.mean(hist[perp]))/float(np.mean(hist[para])+np.mean(hist[perp]))
   
    s_xi=2.*np.sqrt((np.mean(hist[para])*np.std(hist[perp]))**2+(np.mean(hist[perp])*np.std(hist[para]))**2)/(np.mean(hist[para])+np.mean(hist[perp]))**2  
 
    return xi, s_xi

# ===================================================================================================
def roparameter(phi, hist, s_phi=20.):
    # Calculate the relative orientation parameter $\xi$ as defined in Planck intermediate results. XXXV. A&A 586A (2016) 138P.
    #
    # INPUTS
    # phi     - vector with the reference values for the histogram
    # hist    - histogram of relative orientations 
    # s_phi   - range for the definitions of parallel (0 < phi < s_phi and 180-s_phi < phi < 180) or 
    #           perpendicular (90-s_phi < phi < 90+s_phi)
    # OUTPUTS
    # xi          - relative orientation parameter

    perp=(np.abs(phi) > 90.-s_phi).nonzero()
    para=(np.abs(phi) < s_phi).nonzero()
    xi=float(np.mean(hist[para])-np.mean(hist[perp]))/float(np.mean(hist[para])+np.mean(hist[perp]))
  
    s_xi=2.*np.sqrt((np.mean(hist[para])*np.std(hist[perp]))**2+(np.mean(hist[perp])*np.std(hist[para]))**2)/(np.mean(hist[para])+np.mean(hist[perp]))**2

    return xi, s_xi


# ===================================================================================================
def projRS(angles, w=None):
    # Calculate the projected Rayleight statistics as defined in Jow, et al. MNRAS (2018) in press.
    #
    # INPUTS
    # phi      - relative orientation angles defined between -pi/2 and pi/2
    # w        - 
    # OUTPUTS
    # psr      - projected Rayleigh statistic
    # s_prs    - 

    if w is None:
        w = np.ones_like(angles)
    assert w.shape == angles.shape, "Dimensions of phi and wgts must match"

    Zx=circ.tests.vtest(angles, 0., w=w)[1]
    Zy=circ.tests.vtest(angles, np.pi/2., w=w)[1]

    temp=np.sum(w*np.cos(angles)**2)
    s_Zx=np.sqrt((2.*temp-Zx**2)/np.sum(w))

    temp=np.sum(w*np.sin(angles)**2)
    s_Zy=np.sqrt((2.*temp-Zy**2)/np.sum(w))

    x=np.sum(w*np.cos(angles))/np.sum(w)
    y=np.sum(w*np.sin(angles))/np.sum(w)
    mrl=circ.descriptive.resultant_vector_length(angles, w=w)
    s_mrl=np.sqrt(np.sum(w*(np.cos(angles)-x)**2)+np.sum(w*(np.sin(angles)-y)**2))/np.sqrt(np.sum(w))

    meanphi=np.arctan2(y,x)
    s_meanphi=np.sqrt(circ.descriptive.var(angles, w=w))
    #s_meanphi=np.sqrt(Zx*Zx*s_Zy*s_Zy+Zy*Zy*s_Zx*s_Zx)/(Zx**2+Zy**2) 
    # using astropy
    #meanphi[i]=circstats.circmean(angles, weights=w[good])
    #s_meanphi=np.sqrt(circstats.circvar(angles, weights=w))

    return {'Zx': Zx, 's_Zx': s_Zx, 'Zy': Zy, 's_Zy': Zy, 'meanphi': meanphi, 's_meanphi': s_meanphi, 'r': mrl}

# ===================================================================================================
def hroLITE(Imap, Qmap, Umap, steps=10, hsize=15, minI=None, mask=0, ksz=1, showplots=False, w=None, convention='Planck', outh=[0,4,9], savefig=False, prefix='', segmap=None, debug=False):
   # Calculates the relative orientation angle between the density structures and the magnetic field.
   # INPUTS
   # Imap - Intensity or column density map
   # Qmap - Stokes Q map
   # Umap - Stokes U map
   # mask -     

   sz=np.shape(Imap)

   if w is None:
      w=np.ones_like(Imap)
   assert w.shape == Imap.shape, "Dimensions of Imap and w must match"

   # Calculation of relative orientation angles
   phi=roangles(Imap, Qmap, Umap, mask=mask, ksz=ksz, convention=convention, debug=debug)

   if segmap is None:
      segmap=Imap.copy()
   assert Imap.shape == segmap.shape, "Dimensions of Imap and segmap must match" 

   if np.array_equal(np.shape(Imap), np.shape(mask)):
      bad=np.isnan(segmap).nonzero()
      mask[bad]=0.
   else:
      mask=np.ones_like(Imap)
      bad=np.isnan(segmap).nonzero()
      mask[bad]=0.

   if minI is None:
      minI=np.nanmin(segmap)
   bad=(segmap <= minI).nonzero()
   mask[bad]=0.
   bad=np.isnan(phi).nonzero()
   mask[bad]=0.
   segmap[bad]=np.nan 
   bad=np.isnan(segmap).nonzero()
   mask[bad]=0.

   good=(mask > 0.).nonzero()
   hist, bin_edges = np.histogram(segmap[good], bins=int(0.75*np.size(Imap)))     
   bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
   chist=np.cumsum(hist)
   pitch=np.max(chist)/float(steps)

   hsteps=pitch*np.arange(0,steps+1,1)	
   Isteps=np.zeros(steps+1)

   for i in range(0, np.size(Isteps)-1):
      good=np.logical_and(chist>hsteps[i], chist<=hsteps[i+1]).nonzero()
      Isteps[i]=np.min(bin_centre[good])	

   Isteps[np.size(Isteps)-1]=np.nanmax(segmap)

   # Preparing output of the HRO
   hros=np.zeros([steps,hsize])
   s_hros=np.zeros([steps,hsize])
	
   Smap=np.nan*Imap
   xi=np.zeros(steps)
   s_xi=np.zeros(steps)
   Zx=np.zeros(steps)
   Zy=np.zeros(steps)
   s_Zx=np.zeros(steps)
   s_Zy=np.zeros(steps)
   meanphi=np.zeros(steps)
   s_meanphi=np.zeros(steps)
   mrl=np.zeros(steps)
   cdens=np.zeros(steps)

   for i in range(0, np.size(Isteps)-1):

      good=np.logical_and(segmap > Isteps[i], segmap <= Isteps[i+1]).nonzero()
      print(np.size(good))   
 
      hist, bin_edges = np.histogram((180/np.pi)*phi[good], bins=hsize, range=(-90.,90.))	
      bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

      hros[i,:]=hist
      cdens[i]=np.mean([Isteps[i],Isteps[i+1]])
      Smap[good]=Isteps[i]

      TEMPxi, TEMPs_xi = roparameter(bin_centre, hist)
      xi[i]=TEMPxi
      s_xi[i]=TEMPs_xi

      outprojRS = projRS(2.*phi[good], w=w[good])     
      Zx[i]=outprojRS['Zx']
      Zy[i]=outprojRS['Zy']
      s_Zx[i]=outprojRS['s_Zx']
      s_Zy[i]=outprojRS['s_Zy']
      meanphi[i]=0.5*outprojRS['meanphi']
      s_meanphi[i]=0.5*outprojRS['s_meanphi']
      mrl[i]=outprojRS['r']

 
   return {'csteps': Isteps, 'xi': xi, 's_xi': s_xi, 'Zx': Zx, 's_Zx': s_Zx, 'Zy': Zy, 's_Zy': s_Zy, 'meanphi': meanphi, 's_meanphi': s_meanphi, 'asteps': bin_centre, 'hros': hros, 's_hros': s_hros, 'mrl': mrl, 'Smap': Smap, 'Amap': phi} 


# ==================================================================================================
def hro(Imap, Qmap, Umap, steps=10, hsize=15, minI=0., mask=0, ksz=1, w=None, convention='Planck', sigmaQQ=None, sigmaUU=None, mcflag=None, nruns=10, errorbar=None, segmap=None, debug=False):

   #if (convention=='Planck'):
   #   Qmap0=Qmap
   #   Umap0=Umap
   #else:
   #   Qmap0=Qmap
   #   Umap0=-1.*Umap

   if np.logical_or(np.logical_and(sigmaQQ is None, sigmaUU is None), mcflag):

      output0 = hroLITE(Imap, Qmap0, Umap0, steps=steps, hsize=hsize, minI=minI, mask=mask, ksz=ksz, w=w, convention=convention, segmap=segmap, debug=debug)
      isteps=output0['csteps']     
      asteps=output0['asteps'] 
      hros=output0['hros']
      shros=output0['s_hros']   

      zeta=output0['xi']      #roms[0]
      Zx =output0['Zx']     #roms[1]
      Zy =output0['Zy'] 
      mphi=output0['meanphi'] #roms[2]
  
      s_zeta=output0['s_xi']      #sroms[0]
      s_Zx =output0['s_Zx']     #sroms[1]
      s_Zy =output0['s_Zy']
      s_mphi=output0['s_meanphi'] #sroms[2]    
 
      Amap=output0['Amap']   
   
   else: 

      assert Qmap.shape == sigmaQQ.shape, "Dimensions of Qmap and sigmaQQ must match"
      assert Umap.shape == sigmaUU.shape, "Dimensions of Umap and sigmaUU must match"

      hrosvec=np.zeros([nruns,steps,hsize])

      zetavecMC=np.zeros([nruns,steps])
      ZxvecMC=np.zeros([nruns,steps])
      ZyvecMC=np.zeros([nruns,steps])
      mphivecMC=np.zeros([nruns,steps])
      mrlMC=np.zeros([nruns,steps])
   
      s_zetavecMC=np.zeros([nruns,steps])
      s_ZxvecMC=np.zeros([nruns,steps])
      s_ZyvecMC=np.zeros([nruns,steps])
      s_mphivecMC=np.zeros([nruns,steps])
      s_mrlMC=np.zeros([nruns,steps])      

      Acube=np.zeros([nruns,Qmap.shape[0],Qmap.shape[1]])
 
      pbar = tqdm(total=nruns)      

      for i in range(0,nruns):

         QmapR=np.random.normal(loc=Qmap0, scale=sigmaQQ)
         UmapR=np.random.normal(loc=Umap0, scale=sigmaUU)
         hrooutput= hroLITE(Imap, QmapR, UmapR, steps=steps, hsize=hsize, minI=minI, mask=mask, ksz=ksz, w=w, convention=convention, segmap=segmap, debug=False)

         zetavecMC[i,:]=hrooutput['xi']
         ZxvecMC[i,:] =hrooutput['Zx']
         ZyvecMC[i,:] =hrooutput['Zy']
         mphivecMC[i,:]=hrooutput['meanphi']
         mrlMC[i,:]=hrooutput['mrl']

         s_zetavecMC[i,:]=hrooutput['s_xi']
         s_ZxvecMC[i,:] =hrooutput['s_Zx']
         s_ZyvecMC[i,:] =hrooutput['s_Zy']
         s_mphivecMC[i,:]=hrooutput['s_meanphi']

         hrosvec[i,:,:]=hrooutput['hros']

         Acube[i,:,:]=hrooutput['Amap']
     
         pbar.update()

      pbar.close()  

      zeta=zetavecMC.mean(axis=0)
      Zx=ZxvecMC.mean(axis=0)
      Zy=ZyvecMC.mean(axis=0)
      meanphi=circstats.circmean(mphivecMC, axis=0)
      mrl=mrlMC.mean(axis=0)

      Amap=circstats.circmean(Acube, axis=0)
 
      if (errorbar=='MC'):
         s_zeta=s_zetavecMC.std(axis=0)
         s_Zx=s_ZxvecMC.std(axis=0)
         s_Zy=s_ZyvecMC.std(axis=0)
         s_meanphi=circ.descriptive.mean(s_mphivecMC, axis=0)
      else:
         s_zeta=hrooutput['s_xi']         #np.max([s_zetavecMC.std(axis=0),hrooutput['s_xi']], axis=0)
         s_Zx=hrooutput['s_Zx']         #np.max([s_prsvecMC.std(axis=0),hrooutput['s_prs']], axis=0)
         s_Zy=hrooutput['s_Zy']
         s_meanphi=hrooutput['s_meanphi'] #np.max([circ.descriptive.mean(s_mphivecMC, axis=0),hrooutput['s_meanphi']], axis=0)
      s_mrl=mrlMC.std(axis=0)            

      csteps=hrooutput['csteps']
      asteps=hrooutput['asteps']
      Smap=hrooutput['Smap']

      outhros=hrosvec.mean(axis=0)
      s_outhros=hrosvec.std(axis=0) 

   return {'csteps': csteps, 'xi': zeta, 's_xi': s_zeta, 'Zx': Zx, 's_Zx': s_Zx, 'Zy': Zy, 's_Zy': s_Zy, 'meanphi': meanphi, 's_meanphi': s_meanphi, 'asteps': asteps, 'hros': outhros, 's_hros': s_outhros, 'mrl': mrl, 's_mrl': s_mrl, 'Smap': Smap, 'Amap': Amap}

# ===================================================================================================
def hroplts(csteps, roms, sroms, asteps, hros, s_hros, saveplots=False, prefix='', outh=None):

   bin_centres= 0.5*(csteps[0:np.size(csteps)-1]+csteps[1:np.size(csteps)])

   nhist=np.size(bin_centres)
   if outh is None:
      outh=[0, int(nhist/2.), nhist-1] 
   if outh=='all':
      outh=np.arange(nhist)
   color=iter(cm.rainbow(np.linspace(0, 1, np.size(outh))))

   fig = plt.figure(figsize=(4.0, 4.0), dpi=150)
   plt.rc('font', size=8)
   for i in range(0, np.size(outh)):
      c=next(color)
      #labeltext="%.2f"%Isteps[outh[i]] + r' < $N_{\rm H}/$cm$^{-2}$ < ' + "%.2f"%Isteps[outh[i]+1]
      labeltext=str(np.round(csteps[outh[i]],2)) + r' < $N_{\rm H}/$cm$^{-2}$ < ' + str(np.round(csteps[outh[i]+1],2))  
      plt.plot(asteps, hros[outh[i],:], '-', linewidth=2, c=c, label=labeltext) #drawstyle
      plt.errorbar(asteps, hros[outh[i],:], yerr=s_hros[0], c=c)
      plt.xlabel(r'$\phi$')
      plt.ylabel('Histogram density')
      plt.legend()
   if (saveplots):
      plt.savefig(prefix+'_HROs.png', bbox_inches='tight')
      plt.close()
   else:
      plt.show()

   # --------------------------------------------------------------    
   fig = plt.figure(figsize=(4.0, 2.5), dpi=150)
   plt.rc('font', size=8)
   #plt.xlim(csteps[0],csteps[np.size(csteps)-1])
   #plt.plot(bin_centres, roms[0], linewidth=0.5, c='orange', marker='.')
   plt.errorbar(bin_centres, roms[0], yerr=sroms[0], c='orange')
   plt.axhline(y=0., c='k', ls='--')
   plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
   plt.ylabel(r'$\zeta$')
   if (saveplots):
      plt.savefig(prefix+'_zeta.png', bbox_inches='tight')
      plt.close()
   else:
      plt.show()

   # --------------------------------------------------------------
   fig = plt.figure(figsize=(4.0, 2.5), dpi=150)
   plt.rc('font', size=8)
   #plt.xlim(csteps[0],csteps[np.size(csteps)-1])
   #plt.plot(bin_centres, roms[1], linewidth=0.5, c='cyan', marker='.')
   plt.errorbar(bin_centres, roms[1], yerr=sroms[1], c='cyan')
   plt.axhline(y=0., c='k', ls='--')
   plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
   plt.ylabel(r'$V$')
   if (saveplots):
      plt.savefig(prefix+'_PRS.png', bbox_inches='tight')    
      plt.close()
   else:
      plt.show()
   
   # --------------------------------------------------------------
   fig = plt.figure(figsize=(4.0, 2.5), dpi=150)
   plt.rc('font', size=8)
   #plt.xlim(csteps[0],csteps[np.size(csteps)-1])
   #plt.plot(bin_centres, (180./np.pi)*np.abs(roms[2]), linewidth=0.5, c='magenta', marker='.')
   plt.errorbar(bin_centres, (180./np.pi)*np.abs(roms[2]), yerr=(180./np.pi)*sroms[2], c='magenta')
   plt.axhline(y=0., c='k', ls='--')
   plt.axhline(y=90., c='k', ls='--')
   plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
   plt.ylabel(r'$\phi$')
   if (saveplots):
      plt.savefig(prefix+'_meanphi.png', bbox_inches='tight')
      plt.close()
   else:
      plt.show()


