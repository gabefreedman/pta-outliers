#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:09:02 2020

@author: marvin
"""


import numpy as np
import scipy.linalg as sl
from collections import OrderedDict

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter

import libstempo as lt

import piccard as pic
from piccard.jitterext import cython_Uj

from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils

from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix
from enterprise.signals.utils import quant2ind


class ptaLikelihood(object):
    
    def __init__(self, parfile, timfile):
        self.parfile = parfile
        self.timfile = timfile
        
        self.ltpsr = lt.tempopulsar(parfile, timfile)
        self.ephem = self.ltpsr.ephemeris
        self.F0 = self.ltpsr['F0'].val
        self.P0 = 1.0 / self.F0
        
        self.psr = self.load_pulsar(parfile, timfile)
        self.pname = self.psr.name
        self.selection = selections.Selection(selections.by_backend)
        # self.selection = selections.Selection(selections.no_selection)
        
        self.efac_sig = None
        self.equad_sig = None
        self.ecorr_sig = None
        self.rn_sig = None
        self.tm_sig = None
        
        self.basepmin = None
        self.basepmax = None
        self.basepstart = None
        
        self.nfreqcomps = 30
        Fmat, self.Ffreqs = createfourierdesignmatrix_red(self.psr.toas)
        self.Fmat = np.zeros_like(Fmat)
        self.Fmat[:, 1::2] = Fmat[:, ::2]
        self.Fmat[:, ::2] = Fmat[:, 1::2]
        self.Fmat, self.Ffreqs = pic.ptafuncs.fourierdesignmatrix(self.psr.toas, 2*self.nfreqcomps)
        
        self.Umat, self.weights = create_quantization_matrix(self.psr.toas)
        self.Uind = quant2ind(self.Umat)
        self.Uindslc = None
        self.setUindslc()
        
        self.Mmat_g = self.gibbs_set_design()
        self.Zmat = None
        self.ZNZ = None
        self.ZNyvec = None
        
        self.Nvec = np.zeros(len(self.psr.toas))
        self.d_Nvec_d_param = dict()
        self.Jvec = np.zeros(self.Umat.shape[1])
        self.d_Jvec_d_param = dict()
        
        self.Phivec = np.zeros(2 * self.nfreqcomps)
        self.d_Phivec_d_param = dict()
        
        
        self.ptaparams = dict()
        
        self.ptadict = dict()
        
        self.signals = dict()
        
        self.loadSignals()
        self.setBounds()
        self.setZmat()
        self.setFunnelAuxiliary()
        
    
    
    def load_pulsar(self, parfile, timfile):
        toas = np.double(np.array(self.ltpsr.toas())-53000.0)*86400
        residuals = np.double(self.ltpsr.residuals())
        toaerrs = np.double(1e-6*self.ltpsr.toaerrs)
        
        if 'f' in self.ltpsr.flags():
            flags = self.ltpsr.flagvals('f')
        else:
            flags = self.ltpsr.flagvals('be')
        
        isort, iisort = self.argsortTOAs(toas, flags)
        
        psr = Pulsar(parfile, timfile, ephem=self.ephem, sort=False)
        psr._isort = isort
        
        psr._toas = toas
        psr._residuals = residuals
        psr._toaerrs = toaerrs
        
        
        return psr
    
    
    def argsortTOAs(self, toas, flags):
        U, v = create_quantization_matrix(toas, nmin=1)
        isort = np.argsort(toas, kind='mergesort')
        flagvals = list(set(flags))
        
        for cc, col in enumerate(U.T):
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
            

    
    
    def loadSignals(self, incEfac=True, incEquad=True, incEcorr=True, incRn=True, incOut=True,
                    incTiming=True, incFourier=True, incJitter=True):
        if incEfac:
            self.signals.update(self.add_efac())
        if incEquad:
            self.signals.update(self.add_equad())
        if incEcorr:
            self.signals.update(self.add_ecorr())
        if incRn:
            self.signals.update(self.add_rn())
        if incOut:
            self.signals.update(self.add_outlier())
        if incTiming:
            self.signals.update(self.add_timingmodel())
        if incFourier:
            self.signals.update(self.add_fourier())
        if incJitter:
            self.signals.update(self.add_jitter())
        
        index = 0
        for key, sig in self.signals.items():
            sig['pmin'] = np.array(sig['pmin'])
            sig['pmax'] = np.array(sig['pmax'])
            sig['pstart'] = np.array(sig['pstart'])
            sig['index'] = index
            sig['msk'] = slice(sig['index'], sig['index']+sig['numpars'])
            index += sig['numpars']
        
        for key, sig in self.signals.items():
            if sig['type'] == 'rn':
                self.ptaparams.update(dict(zip(sig['name'], sig['pstart'])))
            else:
                self.ptaparams[sig['name']] = sig['pstart'][0]
    
        for ii, key in enumerate(self.ptaparams.keys()):
            if key not in ['timingmodel', 'fouriermode', 'jittermode']:
                self.ptadict[key] = ii
                
                
    def gibbs_set_design(self, gibbsmodel=['rednoise', 'design', 'jitter']):
        F_list = ['Offset', \
                'LAMBDA', 'BETA', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
                'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'TASC', 'EPS1', 'EPS2', \
                'XDOT', 'PBDOT', 'KOM', 'KIN', 'EDOT', 'MTOT', 'SHAPMAX', \
                'GAMMA', 'X2DOT', 'XPBDOT', 'E2DOT', 'OM_1', 'A1_1', 'XOMDOT', \
                'PMLAMBDA', 'PMBETA', 'PX', 'PB', 'A1', 'E', 'ECC', \
                'T0', 'OM', 'OMDOT', 'SINI', 'A1', 'M2']
        F_front_list = ['JUMP', 'F']
        D_list = ['DM', 'DM1', 'DM2', 'DM3', 'DM4']
        U_list = []

        Mmask_F = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmask_D = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmask_U = np.array([0]*len(self.psr.fitpars), dtype=np.bool)
        Mmat_g = np.zeros(self.psr.Mmat.shape)
        for ii, par in enumerate(self.psr.fitpars):
            incrn = False
            for par_front in F_front_list:
                if par[:len(par_front)] == par_front:
                    incrn = True

            if (par in F_list or incrn) and 'rednoise' in gibbsmodel:
                Mmask_F[ii] = True

            if par in D_list and 'dm' in gibbsmodel:
                Mmask_D[ii] = True

            if par in U_list and 'jitter' in gibbsmodel:
                Mmask_U[ii] = True

        if np.sum(np.logical_and(Mmask_F, Mmask_D)) > 0 or \
                np.sum(np.logical_and(Mmask_F, Mmask_U)) > 0 or \
                np.sum(np.logical_and(Mmask_D, Mmask_U)) > 0:
            raise ValueError("Conditional lists cannot overlap")


        Mmask_M = np.array([1]*Mmat_g.shape[1], dtype=np.bool)
        if 'rednoise' in gibbsmodel:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_F))
        if 'dm' in gibbsmodel:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_D))
        if 'jitter' in gibbsmodel:
            Mmask_M = np.logical_and(Mmask_M, \
                    np.logical_not(Mmask_U))

        # Create orthogonals for all of these
        if np.sum(Mmask_F) > 0:
            U, s, Vt = sl.svd(self.psr.Mmat[:,Mmask_F], full_matrices=False)
            Mmat_g[:, Mmask_F] = U

        if np.sum(Mmask_D) > 0:
            U, s, Vt = sl.svd(self.psr.Mmat[:,Mmask_D], full_matrices=False)
            Mmat_g[:, Mmask_D] = U

        if np.sum(Mmask_U) > 0:
            U, s, Vt = sl.svd(self.Mmat[:,Mmask_U], full_matrices=False)
            Mmat_g[:, Mmask_U] = U

        if np.sum(Mmask_M) > 0:
            U, s, Vt = sl.svd(self.psr.Mmat[:,Mmask_M], full_matrices=False)
            Mmat_g[:, Mmask_M] = U
        
        return Mmat_g
    
    
    def setZmat(self):
        if 'timingmodel' in self.ptaparams.keys():
            Zmat = self.Mmat_g.copy()
        if 'fouriermode' in self.ptaparams.keys():
            Zmat = np.append(Zmat, self.Fmat, axis=1)
        if 'jittermode' in self.ptaparams.keys():
            Zmat = np.append(Zmat, self.Umat, axis=1)
        
        self.Zmat = Zmat
    
    
    def setUindslc(self):
        Uinds = []
        for ind in self.Uind:
            Uinds.append((ind.start, ind.stop))
        
        self.smallepochs = []
        for ii, elem in enumerate(self.Uind):
            if elem.stop - elem.start < 4:
                self.smallepochs.append(ii)
        
        Uindslc = np.array(Uinds, dtype=np.int)
        Umatslc = np.delete(self.Umat, self.smallepochs, axis=1)
        self.Uindslc = np.delete(Uindslc, self.smallepochs, axis=0)
        self.Umat = Umatslc
    
    
    def setFunnelAuxiliary(self):
        Nvec = self.psr.toaerrs**2
        ZNyvec = np.dot(self.Zmat.T, self.psr.residuals / Nvec)
        ZNZ = np.dot(self.Zmat.T / Nvec, self.Zmat)
        
        self.ZNZ = ZNZ
        self.ZNyvec = ZNyvec
    
    
    def setBounds(self):
        pmin = []
        pmax = []
        pstart = []
        for key, sig in self.signals.items():
            pmin.extend(sig['pmin'])
            pmax.extend(sig['pmax'])
            pstart.extend(sig['pstart'])
        
        self.basepmin = np.array(pmin)
        self.basepmax = np.array(pmax)
        self.basepstart = np.array(pstart)
    
    
    def updateParams(self, parameters):
        for key, value in self.ptadict.items():
            self.ptaparams[key] = parameters[value]
    
    
    def add_efac(self):
        efac_dct = dict()
        efac = parameter.Uniform(0.001, 5.0)
        ef = white_signals.MeasurementNoise(efac=efac, selection=self.selection)
        self.efac_sig = ef(self.psr)
        for ii, param in enumerate(self.efac_sig.param_names):
            Nvec = self.psr.toaerrs**2 * self.efac_sig._masks[ii]
            newsignal = OrderedDict({'type': 'efac',
                                     'name': param,
                                     'pmin': [0.001],
                                     'pmax': [5.0],
                                     'pstart': [1.0],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Nvec': Nvec})
            efac_dct.update({param : newsignal})
        
        return efac_dct
    
    
    def add_equad(self):
        equad_dct = dict()
        equad = parameter.Uniform(-10.0, -4.0)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=self.selection)
        self.equad_sig = eq(self.psr)
        for ii, param in enumerate(self.equad_sig.param_names):
            Nvec = np.ones_like(self.psr.toaerrs) * self.equad_sig._masks[ii]
            newsignal = OrderedDict({'type': 'equad',
                                     'name': param,
                                     'pmin': [-10.0],
                                     'pmax': [-4.0],
                                     'pstart': [-6.5],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Nvec': Nvec})
            equad_dct.update({param : newsignal})
        
        return equad_dct
    
    
    def add_ecorr(self):
        ecorr_dct = dict()
        ecorr = parameter.Uniform(-10.0, -4.0)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=self.selection)
        self.ecorr_sig = ec(self.psr)
        for ii, param in enumerate(self.ecorr_sig.param_names):
            Nvec = np.ones_like(self.psr.toaerrs) * self.ecorr_sig._masks[ii]
            Jvec = np.array(np.sum(Nvec * self.Umat.T, axis=1) > 0.0, dtype=np.double)
            newsignal = OrderedDict({'type': 'ecorr',
                                     'name': param,
                                     'pmin': [-10.0],
                                     'pmax': [-4.0],
                                     'pstart': [-6.5],
                                     'interval': [True],
                                     'numpars': 1,
                                     'Jvec': Jvec})
            ecorr_dct.update({param : newsignal})
        
        return ecorr_dct
    
    
    def add_rn(self):
        log10_A = parameter.Uniform(-20.0, -10.0)
        gamma = parameter.Uniform(0.02, 6.98)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl, components=self.nfreqcomps, name='rn')
        newsignal = OrderedDict({'type': 'rn',
                                 'name': [self.pname + '_rn_log10_A', self.pname + '_rn_gamma'],
                                 'pmin': [-20.0, 0.02],
                                 'pmax': [-10.0, 6.98],
                                 'pstart': [-14.5, 3.51],
                                 'interval': [True, True],
                                 'numpars': 2})
        self.rn_sig = rn(self.psr)
        return {'rn': newsignal}
    
    
    def add_timingmodel(self):
        tm = gp_signals.TimingModel(use_svd=False)
        npars = self.psr.Mmat.shape[1]
        newsignal = OrderedDict({'type': 'timingmodel',
                                 'name': 'timingmodel',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-10]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})
        self.tm_sig = tm(self.psr)
        return {'timingmodel': newsignal}
    
    
    def add_fourier(self):
        npars = 2 * self.nfreqcomps
        newsignal = OrderedDict({'type': 'fouriermode',
                                 'name': 'fouriermode',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-9]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})
        
        return {'fouriermode': newsignal}
    
    
    def add_jitter(self):
        npars = len(self.Jvec)
        newsignal = OrderedDict({'type': 'jittermode',
                                 'name': 'jittermode',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-9]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})
        
        return {'jittermode': newsignal}
        
    
    
    def add_outlier(self):
        newsignal = OrderedDict({'type': 'outlier',
                                 'name': self.pname + '_outlierprob',
                                 'pmin': [0.0],
                                 'pmax': [1.0],
                                 'pstart': [0.001],
                                 'interval': [True],
                                 'numpars': 1})
        return {'outlier': newsignal}
    
    
    def setWhiteNoise(self, calc_gradient=True):
        self.Nvec[:] = 0
        self.Jvec[:] = 0
        
        ef = self.efac_sig
        eq = self.equad_sig
        ec = self.ecorr_sig
        
        self.Nvec[:] = ef.get_ndiag(self.ptaparams) + eq.get_ndiag(self.ptaparams)
        fullJvec = ec.get_phi(self.ptaparams)
        self.Jvec[:] = np.delete(fullJvec, self.smallepochs, axis=0)
        
        if calc_gradient:
            if ef:
                for param in ef.param_names:
                    self.d_Nvec_d_param[self.ptadict[param]] = 2 * \
                                                               self.signals[param]['Nvec'] * \
                                                               self.ptaparams[param]
            if eq:
                for param in eq.param_names:
                    self.d_Nvec_d_param[self.ptadict[param]] = self.signals[param]['Nvec'] * \
                                                               2 * np.log(10) * \
                                                               10**(2*self.ptaparams[param])
            if ec:
                for param in ec.param_names:
                    self.d_Jvec_d_param[self.ptadict[param]] = self.signals[param]['Jvec'] * \
                                                               2 * np.log(10) * \
                                                               10**(2*self.ptaparams[param])
        return
        
        
        
    def setPhi(self, calc_gradient=True):
        self.Phivec[:] = 0
        
        # rn = self.rn_sig
        log10A = self.ptaparams[self.pname + '_rn_log10_A']
        gamma = self.ptaparams[self.pname + '_rn_gamma']
        sTmax = self.psr.toas.max() - self.psr.toas.min()
        
        # self.Phivec[:] = rn.get_phi(self.ptaparams)
        self.Phivec[:] = powerlaw(log10A, gamma, sTmax, self.Ffreqs)
        
        if calc_gradient:
            d_mat = d_powerlaw(log10A, gamma, sTmax, self.Ffreqs)
            for key, param in self.ptaparams.items():
                if key.endswith('log10_A'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:,0]
                elif key.endswith('gamma'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:,1]
        
        return
    
    
    def setOutliers(self):
        for key, param in self.ptaparams.items():
            if key.endswith('outlierprob'):
                self.outlier_prob = param
                self.d_Pb_ind = [self.ptadict[key]]
        
        return
    
    
    def setDetSources(self, parameters, calc_gradient=True):
        d_L_d_b = np.zeros_like(parameters)
        d_Pr_d_b = np.zeros_like(parameters)
        self.outlier_sig_dict = dict()
        
        self.detresiduals = self.psr.residuals.copy()
        
        for key, sig in self.signals.items():
            sparams = parameters[sig['msk']]
            
            if sig['type'] == 'bwm':
                pass
            elif sig['type'] == 'timingmodel':
                self.detresiduals -= np.dot(self.Mmat_g, sparams)
            elif sig['type'] == 'fouriermode':
                self.detresiduals -= np.dot(self.Fmat, sparams)
            elif sig['type'] == 'jittermode':
                self.detresiduals -= cython_Uj(sparams, self.Uindslc, len(self.detresiduals))
        
        if calc_gradient:
            pulsarind = 0
            if pulsarind not in self.outlier_sig_dict:
                self.outlier_sig_dict[pulsarind] = []
            for key, sig in self.signals.items():
                parslice = sig['msk']
                sparams = parameters[parslice]
                
                if sig['type'] == 'bwm':
                    pass
                elif sig['type'] == 'timingmodel':
                    d_L_d_xi = np.zeros(self.Mmat_g.shape[1])
                    
                    d_L_d_b_o = self.Mmat_g.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))
                    
                    d_L_d_b[parslice] = d_L_d_xi
                elif sig['type'] == 'fouriermode':
                    d_L_d_xi = np.zeros(self.Fmat.shape[1])
                    phivec = self.Phivec.copy()
                    
                    d_L_d_b_o = self.Fmat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))
                    
                    d_Pr_d_xi = -sparams / phivec
                    d_L_d_b[parslice] = d_L_d_xi
                    d_Pr_d_b[parslice] = d_Pr_d_xi
                elif sig['type'] == 'jittermode':
                    d_L_d_xi = np.zeros(self.Umat.shape[1])
                    
                    d_L_d_b_o = self.Umat.T * (self.detresiduals / self.Nvec)[None, :]
                    self.outlier_sig_dict[pulsarind].append((parslice, d_L_d_b_o))
                    
                    d_Pr_d_xi = -sparams / self.Jvec
                    d_L_d_b[parslice] = d_L_d_xi
                    d_Pr_d_b[parslice] = d_Pr_d_xi
        
        self.d_L_d_b = d_L_d_b
        self.d_Pr_d_b = d_Pr_d_b
        return
    
    
    def set_hyperparameters(self, parameters, calc_gradient=True):
        self.updateParams(parameters)
        
        self.setPhi(calc_gradient=calc_gradient)
        self.setWhiteNoise(calc_gradient=calc_gradient)
        self.setOutliers()
    
    
    def base_loglikelihood_grad(self, parameters, set_hyper_params=True, calc_gradient=True):
        
        if set_hyper_params:
            self.set_hyperparameters(parameters, calc_gradient=calc_gradient)
            self.setDetSources(parameters, calc_gradient=calc_gradient)
        
        d_L_d_b, d_Pr_d_b = self.d_L_d_b, self.d_Pr_d_b
        gradient = np.zeros_like(d_L_d_b)
        
        bBb = np.zeros_like(0, dtype=float)
        ldB = np.zeros_like(0, dtype=float)
        logl_outlier = np.zeros_like(0, dtype=float)
        
        P0 = self.P0
        Pb = self.outlier_prob
        
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
        
        if 'fouriermode' in self.ptaparams.keys():
            pslc = self.signals['fouriermode']['msk']

            bsqr = parameters[pslc]**2
            phivec = self.Phivec # + Svec[fslc]

            bBb += np.sum(bsqr / phivec)
            ldB += np.sum(np.log(phivec))

            gradient[pslc] += d_Pr_d_b[pslc]

            for key, d_Phivec_d_p in self.d_Phivec_d_param.items():
                gradient[key] += 0.5 * np.sum(bsqr * d_Phivec_d_p / phivec**2)
                gradient[key] -= 0.5 * np.sum(d_Phivec_d_p / phivec)
            
        if 'dmfouriermode' in self.ptaparams.keys():
            pass

        if 'jittermode' in self.ptaparams.keys():
            pslc = self.signals['jittermode']['msk']

            bsqr = parameters[pslc]**2
            jvec = self.Jvec

            bBb += np.sum(bsqr / jvec)
            ldB += np.sum(np.log(jvec))
            
            gradient[pslc] += d_Pr_d_b[pslc]

            for key, d_Jvec_d_p in self.d_Jvec_d_param.items():
                gradient[key] += 0.5 * np.sum(bsqr * d_Jvec_d_p / jvec**2)
                gradient[key] -= 0.5 * np.sum(d_Jvec_d_p / jvec)

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