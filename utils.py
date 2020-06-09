#!/usr/bin/env python

# utils.py

"""
General utilities module containing functions for constructing PTA
covariance matrices, computing matrix products and determinants, and
computing the log likelihood for a given 1-pulsar PTA object.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sps
import scipy.linalg as sl
from sksparse.cholmod import cholesky

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
import enterprise.constants as const
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise.signals.utils import KernelMatrix


def fourier_basis(psr):
    Ti = psr.toas.max() - psr.toas.min()
    fmin = 1/Ti
    fmax = 30/Ti
    f = np.linspace(fmin, fmax, 30)
    ranphase = np.zeros(30)
    
    Ffreqs = np.repeat(f, 2)
    N = len(psr.toas)
    F = np.zeros((N, 2*30))
    
    F[:, ::2] = np.sin(2 * np.pi * psr.toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * psr.toas[:, None] * f[None, :] + ranphase[None, :])
    
    basis = F, Ffreqs
    return basis


def ecorr_prior(signal, params=params, psr=psr):
    return [10**(2*l.value) for l in list(signal._params.values())]


def rn_prior(psr, params, signal=None, components=2):
    _, f = fourier_basis(psr)
    log10_A = params[psr.name+'_rn_log10_A']
    gamma = params[psr.name+'_rn_gamma']
    
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    return (
        (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, components)
    )


def timing_prior(psr, params=None, signal=None):
    weights = np.ones_like(psr.Mmat.shape[1])
    return weights * 1e40


def construct_N(sc, psr):
    # Function to construct white noise matrix `N`
    # For now it only accomadates single pulsar PTAs
    
    # W matrix is a diagonal matrix of TOA uncertainties squared
    W = np.diag(psr.toaerrs) ** 2

    # Separate out different types of white noise
    ef_noise = sc._signals[0]
    eq_noise = sc._signals[1]

    # Let's divide up into our different observing systems (our k's)
    # First we need to generate a 'mask' dictionary, aka matching receivers
    # to the TOAs they measured
    sel = selection(psr)
    masks = sel.masks

    ef_vals = [l.value**2 for l in list(ef_noise._params.values())]
    ef_dict = dict(zip(ef_noise._keys, ef_vals))

    eq_vals = [10**(2*l.value) for l in list(eq_noise._params.values())]
    eq_dict = dict(zip(eq_noise._keys, eq_vals))

    # Now we construct the N_k dictionary, one N matrix per backend+receiver combo
    # Then sum them together to get the final N
    N_k = {}
    for key, mask in masks.items():
        N_k[key] = (ef_dict[key] * W + eq_dict[key]*np.identity(len(psr.toas))) * mask

    N = sum(N_k.values())
    
    return N


def construct_basis(signal, params):
    bases, labels = {}, {}
    for key, mask in zip(signal._keys, signal._masks):
        bases[key], labels[key] = signal._bases[key](params=params, mask=mask)

    nc = sum(F.shape[1] for F in bases.values())
    basis = np.zeros((len(signal._masks[0]), nc))
    slices = {}
    nctot = 0

    for key, mask in zip(signal._keys, signal._masks):
        Fmat = bases[key]
        nn = Fmat.shape[1]
        basis[mask, nctot : nn + nctot] = Fmat
        slices.update({key: slice(nctot, nn+nctot)})
        nctot += nn
    
    return basis, slices


def get_phi(signal, params, prior):
    basis, slices = construct_basis(signal, params)
    
    nc = basis.shape[1]
    phi = KernelMatrix(nc)
    
    priorvals = prior(signal=signal, params=params, psr=psr)
    if len(signal._keys) > 1:
        priordict = dict(zip(signal._keys, priorvals))
    elif len(signal._keys) == 1:
        priordict = {signal._keys[0]: priorvals}
    
    for key, slc, in slices.items():
        ndim = slc.stop - slc.start
        phislc = priordict[key] * np.ones(ndim)
        phi = phi.set(phislc, slc)
    
    return phi


def construct_T(sc, params):
    idxlst = list(sc._idx.values())
    ncol = idxlst[-1][-1] + 1
    nrow = len(sc._residuals)
    T = np.zeros((nrow, ncol))
    for signal in sc._signals:
        if signal in sc._idx:
            basis, _ = construct_basis(signal, params)
            T[:, sc._idx[signal]] = basis
    return T


def construct_phi(sc, params, priors):
    idxlst = list(sc._idx.values())
    ncol = int(idxlst[-1][-1] + 1)
    phi = KernelMatrix(ncol)
    for signal in sc._signals:
        if signal in sc._idx:
            phislc = get_phi(signal, params, priors[signal.name])
            phi = phi.add(phislc, sc._idx[signal])
    return phi


def get_TNr(T, Nvec, r):
    mult = np.array(r/Nvec)
    TNr = np.dot(T.T, mult)
    return TNr


def get_TNT(T, Nvec):
    TNT = np.dot(T.T, np.array(T/Nvec[:, None]))
    return TNT


def get_rT_Ninv_r(r, N):
    rT = np.transpose(r)
    Ninv = np.linalg.inv(N)
    
    rT_Ninv_r = np.dot(rT, np.dot(Ninv, r))
    return rT_Ninv_r


def get_phiinv(phi):
    phiinv = 1.0 / phi
    logdet_phi = np.sum(np.log(phi))
    
    return phiinv, logdet_phi


def custom_likelihood(pta, psr, params):
    ln_likelihood = 0
    
    sc = pta.pulsarmodels[0]
    priordict = {psr.name + '_basis_ecorr': ecorr_prior,
                 psr.name + '_rn': rn_prior,
                 psr.name + '_linear_timing_model': timing_prior}
    
    r = psr.residuals
    
    N = construct_N(sc, psr)
    Nvec = np.diag(N)
    logdet_N = np.sum(np.log(Nvec))
    
    T = construct_T(sc, params)
    
    phi = construct_phi(sc, params, priordict)
    phiinv, logdet_phi = get_phiinv(phi)

    TNr = get_TNr(T, Nvec, r)
    TNT = get_TNT(T, Nvec)
    rT_Ninv_r = get_rT_Ninv_r(r, N)
    
    Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
    cf = sl.cho_factor(Sigma)
    expval = sl.cho_solve(cf, TNr)
    logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))
    
    ln_likelihood += -0.5 * (rT_Ninv_r + logdet_N)
    ln_likelihood += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)
    return ln_likelihood

