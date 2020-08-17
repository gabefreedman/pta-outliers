#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:17:29 2020

@author: marvin
"""

import numpy as np

from enterprise.signals.utils import create_quantization_matrix


def d_powerlaw(lAmp, Si, Tmax, freqs, ntotfreqs=None, nfreqind=None, spy=31557600.0):
    if ntotfreqs is None:
        ntotfreqs = len(freqs)
    if nfreqind is None:
        nfreqind = 0

    freqpy = freqs * spy
    d_mat = np.zeros((ntotfreqs, 3))

    d_mat[nfreqind:nfreqind+len(freqs), 0] = (2*np.log(10)*10**(2*lAmp) * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[nfreqind:nfreqind+len(freqs), 1] = -np.log(freqpy)*(10**(2*lAmp) * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[nfreqind:nfreqind+len(freqs), 2] = 0.0

    return d_mat


def argsortTOAs(toas, flags):
    U, _ = create_quantization_matrix(toas, nmin=1)
    isort = np.argsort(toas, kind='mergesort')
    flagvals = list(set(flags))

    for _, col in enumerate(U.T):
        for flag in flagvals:
            flagmask = (flags[isort] == flag)
            if np.sum(col[isort][flagmask]) > 1:
                colmask = col[isort].astype(np.bool)
                epmsk = flagmask[colmask]
                epinds = np.flatnonzero(epmsk)
                if len(epinds) == epinds[-1] - epinds[0] + 1:
                    # Keys are exclusively in succession
                    pass
                else:
                    episort = np.argsort(flagmask[colmask], kind='mergesort')
                    isort[colmask] = isort[colmask][episort]
            else:
                # Only one element, always ok
                pass
    # Now that we have a correct permutation, also construct the inverse
    iisort = np.zeros(len(isort), dtype=np.int)
    for ii, p in enumerate(isort):
        iisort[p] = ii

    return isort, iisort

def set_Uindslc(Umat, Uind):
    Uinds = []
    for ind in Uind:
        Uinds.append((ind.start, ind.stop))

    smallepochs = []
    for ii, elem in enumerate(Uind):
        if elem.stop - elem.start < 4:
            smallepochs.append(ii)

    Uindslc = np.array(Uinds, dtype=np.int)
    Umatslc = np.delete(Umat, smallepochs, axis=1)
    
    Uindslc = np.delete(Uindslc, smallepochs, axis=0)
    Umat = Umatslc
    
    return Umat, Uindslc