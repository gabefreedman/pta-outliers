#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:38:53 2020

@author: marvin

An implementation of the base, single PTA likelihood class
"""

import numpy as np

from piccard.jitterext import cython_Uj


class baseLikelihood(object):

    def __init__(self, likob):

        self.likob = likob
        self.npsrs = len(self.likob.ptapsrs)
        self.psr = likob.ptapsrs[0]
        self.ptasignals = likob.ptasignals
        self.pardes = likob.pardes
        self.dimensions = likob.dimensions
        self.msk = likob.interval

        self.npf = likob.npf
        self.npfdm = likob.npfdm
        self.npu = [len(self.psr.avetoas)]
        self.npm = self.psr.Mmat.shape[1]

        self.detresiduals = None

        self.Phivec = np.zeros(np.sum(self.npf))
        self.d_Phivec_d_param = dict()
        self.Thetavec = np.zeros(np.sum(self.npfdm))
        self.d_Theta_d_param = dict()
        self.Svec = np.zeros(np.max(self.npf))

        self.Nvec = np.zeros(len(self.psr.toas))
        self.d_Nvec_d_param = dict()
        self.Jvec = np.zeros(len(self.psr.avetoas))
        self.d_Jvec_d_param = dict()

        self.outlier_prob = 0.0
        self.d_Pb_ind = []
        self.outlier_sig_dict = dict()

        self.rGr = np.zeros_like(self.npsrs)
        self.GNGldet = np.zeros_like(self.npsrs)
        
        self.basepmin = likob.likob.likob.pmin
        self.basepmax = likob.likob.likob.pmax
        self.basepstart = likob.likob.likob.pstart
        

        self.clearLikob()


    def clearLikob(self):
        # Frees up memory by clearing piccard likelihood object
        self.likob = None


    def constructPhi(self, parameters, calc_gradient=True):
        self.Phivec[:] = 0
        self.Thetavec[:] = 0

        selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        if not hasattr(self, 'npff'):
            npff = [None]
        else:
            npff = self.npff

        for ss in np.hstack((self.psr.sig_Phi_inds,
                             self.psr.sig_Theta_inds,
                             self.psr.sig_Beta_inds)):
            signal = self.ptasignals[int(ss)]

            if selection[int(ss)]:
                sparameters = signal['pstart'].copy()
                sparameters[signal['bvary']] = parameters[signal['parindex']:
                                                          signal['parindex']+
                                                          signal['npars']]


            if signal['stype'] == 'powerlaw':
                # lAmp = log10 Amplitude
                lAmp = sparameters[0]
                # Si = spectral index (gamma)
                Si = sparameters[1]
                sTmax = signal['Tmax']
                sfreqs = self.psr.Ffreqs

                if signal['corr'] == 'single':
                    findex = int(np.sum(npff[:0]))
                    nfreq = int(self.npf[0]/2)
                    # invoke powerlaw prior
                    pcd = powerlaw(lAmp, Si, sTmax, sfreqs)

                self.Phivec[findex:findex+2*nfreq] += pcd

                if calc_gradient:
                    ntotfreqs = np.sum(npff)
                    d_mat = d_powerlaw(lAmp, Si, sTmax, sfreqs, ntotfreqs=ntotfreqs, nfreqind=findex)
                    for col in range(d_mat.shape[1]):
                        if signal['bvary'][col]:
                            self.d_Phivec_d_param[signal['parindex']+col] = d_mat[:,col]



    def setPsrNoise(self, parameters, calc_gradient=True):
        self.Nvec[:] = 0
        self.Jvec[:] = 0

        selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        for ss in self.psr.sig_NJ_inds:
            signal = self.ptasignals[ss]

            if selection[ss]:
                if signal['stype'] == 'efac':
                    if signal['npars'] == 1:
                        pefac = parameters[signal['parindex']]
                    else:
                        pefac = signal['pstart'][0]

                    self.Nvec += signal['Nvec'] * pefac**2

                    if calc_gradient and signal['npars'] == 1:
                        self.d_Nvec_d_param[signal['parindex']] = 2 * signal['Nvec'] * pefac

                elif signal['stype'] == 'equad':
                    if signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[signal['parindex']])
                    else:
                        pequadsqr = 10**(2*signal['pstart'][0])
    
                    self.Nvec += signal['Nvec'] * pequadsqr
    
                    if calc_gradient and signal['npars'] == 1:
                        self.d_Nvec_d_param[signal['parindex']] = signal['Nvec'] * \
                                                             2*np.log(10)*10**(2*parameters[signal['parindex']])
        return
    
    
    def setPb_outliers(self, parameters):

        selection = np.array([1]*len(self.ptasignals), dtype=np.bool)
    
        for ss in self.psr.sig_Outlier_inds:
            signal = self.ptasignals[ss]
    
            if selection[ss]:
                sparameters = signal['pstart'].copy()
                parslc = np.arange(signal['parindex'], signal['parindex']+signal['npars'])
                sparameters[signal['bvary']] = parameters[parslc]
    
                if signal['stype'] == 'outlierprob':
                    self.outlier_prob = sparameters[0]
                    if np.sum(signal['bvary']) > 0:
                        self.d_Pb_ind = parslc
        return
    
    
    def set_hyperparameters(self, parameters):
        self.constructPhi(parameters)
        self.setPsrNoise(parameters)
        self.setPb_outliers(parameters)
    
    
    def updateDetSources(self, parameters):
        d_L_d_b = np.zeros_like(parameters)
        d_Pr_d_b = np.zeros_like(parameters)
        self.outlier_sig_dict = dict()
    
        selection = np.array([1]*len(self.ptasignals), dtype=np.bool)
        self.detresiduals = self.psr.residuals.copy()
    
        for ss, signal in enumerate(self.ptasignals):
            if selection[ss]:
                sparameters = signal['pstart'].copy()
                sparameters[signal['bvary']] = parameters[signal['parindex']:signal['parindex']+signal['npars']]
    
                if signal['stype'] == 'bwm':
                    pass
                elif signal['stype'] == 'psrbwm':
                    pass
                elif signal['stype'] == 'glitch':
                    pass
                elif signal['stype'] == 'timingmodel_xi':
                    self.detresiduals -= np.dot(self.psr.Mmat_g, sparameters)
                elif signal['stype'] == 'fouriermode_xi':
                    self.detresiduals -= np.dot(self.psr.Fmat, sparameters)
                elif signal['stype'] == 'dmfouriermode_xi':
                    pass
                elif signal['stype'] == 'jittermode_xi':
                    self.detresiduals -= cython_Uj(sparameters, self.psr.Uinds, len(self.detresiduals))
    
        for ss, signal in enumerate(self.ptasignals):
            if selection[ss]:
                sparameters = signal['pstart'].copy()
                sparameters[signal['bvary']] = parameters[signal['parindex']:signal['parindex']+signal['npars']]
                pp = signal['pulsarind']
    
                if signal['stype'] == 'bwm':
                    pass
                elif signal['stype'] == 'psrbwm':
                    pass
                elif signal['stype'] == 'timingmodel_xi':
                    parslice = slice(signal['parindex'], signal['parindex']+signal['npars'])
                    smask = signal['bvary']
    
                    d_L_d_xi = np.zeros(self.psr.Mmat_g.shape[1])
                    if pp not in self.outlier_sig_dict:
                        self.outlier_sig_dict[pp] = []
    
                    d_L_d_b_o = self.psr.Mmat_g.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pp].append((parslice, d_L_d_b_o[smask,:]))
    
                    d_L_d_b[parslice] = d_L_d_xi[smask]
    
                elif signal['stype'] == 'fouriermode_xi':
                    parslice = slice(signal['parindex'], signal['parindex']+signal['npars'])
                    smask = signal['bvary']
                    findex = np.sum(self.npf[:pp])
                    nfs = self.npf[pp]
                    phislice = slice(findex, findex+nfs)
                    phivec = self.Phivec[phislice] + self.Svec[:nfs]
    
                    d_L_d_xi = np.zeros(self.psr.Fmat.shape[1])
                    if pp not in self.outlier_sig_dict:
                        self.outlier_sig_dict[pp] = []
    
                    d_L_d_b_o = self.psr.Fmat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pp].append((parslice, d_L_d_b_o[smask,:]))
    
                    d_Pr_d_xi = -sparameters / phivec
                    d_L_d_b[parslice] = d_L_d_xi[smask]
                    d_Pr_d_b[parslice] = d_Pr_d_xi[smask]
    
                elif signal['stype'] == 'dmfouriermode_xi':
                    pass
                elif signal['stype'] == 'jittermode_xi':
                    parslice = slice(signal['parindex'], signal['parindex']+signal['npars'])
                    smask = signal['bvary']
    
                    d_L_d_xi = np.zeros(self.npu[pp])
                    if not pp in self.outlier_sig_dict:
                        self.outlier_sig_dict[pp] = []
    
                    d_L_d_b_o = self.psr.Umat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pp].append((parslice, d_L_d_b_o[smask,:]))
    
                    d_Pr_d_xi = -sparameters / self.Jvec
                    d_L_d_b[parslice] = d_L_d_xi[smask]
                    d_Pr_d_b[parslice] = d_Pr_d_xi[smask]
        
        self.d_L_d_b = d_L_d_b
        self.d_Pr_d_b = d_Pr_d_b
        return
    
    
    def base_loglikelihood_grad(self, parameters, set_hyper_params=True):

        if set_hyper_params:
            self.set_hyperparameters(parameters)
            
            self.updateDetSources(parameters)
    
        d_L_d_b, d_Pr_d_b = self.d_L_d_b, self.d_Pr_d_b
        gradient = np.zeros_like(d_L_d_b)
    
        bBb = np.zeros_like(self.rGr, dtype=float)
        ldB = np.zeros_like(self.GNGldet, dtype=float)
        logl_outlier = np.zeros_like(self.rGr, dtype=float)
    
        P0 = self.psr.P0
        Pb = self.outlier_prob
    
        if np.sum(self.Jvec) == 0:
            self.rGr = np.sum(self.detresiduals**2 / self.Nvec)
            self.GNGldet = np.sum(np.log(self.Nvec))
    
            lln = self.detresiduals**2 / self.Nvec
            lld = np.log(self.Nvec) + np.log(2*np.pi)
            logL0 = -0.5*lln -0.5*lld
            bigL0 = (1. - Pb) * np.exp(logL0)
            bigL = bigL0 + Pb/P0
            logl_outlier += np.sum(np.log(bigL))

            for pslc, d_L_d_b_o in self.outlier_sig_dict[0]:
                gradient[pslc] += np.sum(d_L_d_b_o * bigL0[None,:]/bigL[None,:], axis=1)

            for pbind in self.d_Pb_ind:
                gradient[pbind] += np.sum((-np.exp(logL0)+1.0/P0)/bigL)

            for key, d_Nvec_d_p in self.d_Nvec_d_param.items():
                d_L_d_b_o = 0.5*(self.detresiduals**2 * d_Nvec_d_p / self.Nvec**2 - d_Nvec_d_p / self.Nvec)
                gradient[key] += np.sum(d_L_d_b_o * bigL0/bigL)

        if self.psr.fourierind is not None:
            findex = np.sum(self.npf[:0])
            nfreq = self.npf[0]
            ind = self.psr.fourierind
            fslc = slice(findex, findex+nfreq)
            pslc = slice(ind, ind+nfreq)

            bsqr = parameters[pslc]**2
            phivec = self.Phivec[fslc] # + Svec[fslc]

            bBb += np.sum(bsqr / phivec)
            ldB += np.sum(np.log(phivec))

            gradient[pslc] += d_Pr_d_b[pslc]

            for key, d_Phivec_d_p in self.d_Phivec_d_param.items():
                gradient[key] += 0.5 * np.sum(bsqr * d_Phivec_d_p / phivec**2)
                gradient[key] -= 0.5 * np.sum(d_Phivec_d_p / phivec)

        if self.psr.dmfourierind is not None:
            pass

        if self.psr.jitterind is not None:
            npus = self.npu[0]
            ind = self.psr.jitterind
            pslc = slice(ind, ind+npus)

            bsqr = parameters[pslc]**2
            jvec = self.Jvec

            bBb += np.sum(bsqr / jvec)
            ldB += np.sum(np.log(jvec))

        ll = np.sum(logl_outlier) - 0.5*np.sum(bBb) - 0.5*np.sum(ldB)

        return ll, gradient


def powerlaw(lAmp, Si, Tmax, freqs, spy=31557600.0):
    freqpy = freqs * spy
    return (10**(2*lAmp) * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy**(-Si)


def d_powerlaw(lAmp, Si, Tmax, freqs, ntotfreqs=None, nfreqind=None, spy=31557600.0):
    if ntotfreqs is None:
        ntotfreqs = len(freqs)
    if nfreqind is None:
        nfreqind = 0

    freqpy = freqs * spy
    d_mat = np.zeros((ntotfreqs, 3))

    d_mat[nfreqind:nfreqind+len(freqs),0] = (2*np.log(10)*10**(2*lAmp) * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[nfreqind:nfreqind+len(freqs),1] = -np.log(freqpy)*(10**(2*lAmp) * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[nfreqind:nfreqind+len(freqs),2] = 0.0

    return d_mat

def d2_powerlaw(lAmp, Si, Tmax, freqs, ntotfreqs=None, nfreqind=None, spy=31557600.0):
    if ntotfreqs is None:
        ntotfreqs = len(freqs)
    if nfreqind is None:
        nfreqind = 0

    freqpy = freqs * spy
    d_mat = np.zeros((ntotfreqs, 3, 3))

    fslc = slice(nfreqind,nfreqind+len(freqs))

    d_mat[fslc,0,0] = ( (2*np.log(10))**2 * 10**(2*lAmp) * spy**3 
            / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[fslc,0,1] = ( -2*np.log(10)*np.log(freqpy) * 10**(2*lAmp) 
            * spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[fslc,0,2] = 0.0

    d_mat[fslc,1,0] = d_mat[nfreqind:nfreqind+len(freqs),0,1]
    d_mat[fslc,1,1] = np.log(freqpy)**2 * (10**(2*lAmp) * spy**3 
            / (12*np.pi*np.pi * Tmax)) * freqpy ** (-Si)
    d_mat[fslc,1,2] = 0.0

    d_mat[fslc,2,0] = 0.0
    d_mat[fslc,2,1] = 0.0
    d_mat[fslc,2,2] = 0.0

    return d_mat
