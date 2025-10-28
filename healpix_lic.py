# This file is part of MAGNETAR, the set of magnetic field analysis tools
#
# Copyright (C) 2013-2025 Juan Diego Soler

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import norm

from tqdm import tqdm

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



