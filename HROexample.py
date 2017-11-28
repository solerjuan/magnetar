# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2017 Juan Diego Soler

import numpy as np
import matplotlib.pyplot as plt

from hro import *
from hro3d import *

Imap=fits.open('data/Taurusfwhm5_logNHmap.fits')
Qmap=fits.open('data/Taurusfwhm10_Qmap.fits')
Umap=fits.open('data/Taurusfwhm10_Umap.fits')

xdens, xi = hro(Imap[0].data, Qmap[0].data, Umap[0].data, steps=10, minI=21.5)

fig=plt.figure()
plt.plot(xdens, xi, '-', linewidth=2, c=c)
plt.axhline(y=0., c='k', ls='--')
plt.xlabel(r'log$_{10}$ ($N_{\rm H}/$cm$^{-2}$)')
plt.ylabel(r'$\zeta$')
plt.show()


