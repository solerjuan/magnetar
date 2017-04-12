import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import Gaussian2DKernel

def roangles(Imap, Qmap, Umap):

	psi=0.5*np.arctan2(-Umap,Qmap)	
	ex=np.sin(psi)
    	ey=np.cos(psi)

    	grad=np.gradient(Imap, edge_order=2)
	#kernel=Gaussian2DKernel(3)
	#grad=np.gradient(convolve_fft(Imap, kernel))	
	phi=np.arctan2(grad[0]*ex-grad[1]*ey, grad[0]*ey+grad[1]*ex)		

	bad=np.logical_or(grad[0]*grad[0]+grad[1]*grad[1]==0., Qmap*Qmap+Umap*Umap==0.).nonzero()	
	phi[bad]=np.sqrt(-1)

    	return np.abs(phi)


def hro(Imap, Qmap, Umap, steps=10, hsize=20, minI=0.):

	sz=np.shape(Imap)
	phi=roangles(Imap, Qmap, Umap)

	hist, bin_edges = np.histogram(Imap[(Imap > minI).nonzero()], bins=100*sz[0])
	bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])

	chist=np.cumsum(hist)
	pitch=np.max(chist)/float(steps)
	
	hsteps=pitch*np.arange(0,steps+1,1)	
	Isteps=np.zeros(steps+1)

	for i in range(0, np.size(Isteps)-1):
		good=np.logical_and(chist>hsteps[i],chist<hsteps[i+1]).nonzero()
		Isteps[i]=np.min(bin_centre[good])	
	Isteps[np.size(Isteps)-1]=np.max(Imap)
	print(Isteps)

	hros=np.zeros([steps,hsize])	
	Smap=0.*Imap

	for i in range(0, np.size(Isteps)-1):
		good=np.logical_and(Imap>Isteps[i],Imap<Isteps[i+1]).nonzero()
		print(np.size(good))
		hist, bin_edges = np.histogram((180/np.pi)*phi[good], bins=hsize, range=(0.,180.))	
		bin_centre=0.5*(bin_edges[0:np.size(bin_edges)-1]+bin_edges[1:np.size(bin_edges)])
		hros[i,:]=hist
		Smap[good]=i
		plt.plot(bin_centre, hist)
	plt.show()	

	import pdb; pdb.set_trace()

        return hros, Isteps


def main(args=None):

    	if res.prnt:
		print('Primes: {0}'.format(primes))

Imap=fits.open('data/Taurusfwhm5_logNHmap.fits')
Qmap=fits.open('data/Taurusfwhm10_Qmap.fits')
Umap=fits.open('data/Taurusfwhm10_Umap.fits')
hro(Imap[0].data, Qmap[0].data, Umap[0].data, minI=21.0)

