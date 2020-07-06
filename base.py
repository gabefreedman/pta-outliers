#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:09:02 2020

@author: marvin
"""


import numpy as np
from collections import OrderedDict

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter

from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils

from enterprise.signals.gp_bases import createfourierdesignmatrix_red


class ptaLikelihood(object):
    
    def __init__(self, parfile, timfile):
        self.parfile = parfile
        self.timfile = timfile
        
        self.psr = self.load_pulsar(parfile, timfile)
        self.pname = self.psr.name
        self.selection = selections.Selection(selections.no_selection)
        
        self.efac_sig = None
        self.equad_sig = None
        self.ecorr_sig = None
        self.rn_sig = None
        self.tm_sig = None
        
        self.nfreqcomps = 30
        self.Fmat, self.Ffreqs = createfourierdesignmatrix_red(self.psr.toas)
        
        self.Nvec = np.zeros(len(self.psr.toas))
        self.d_Nvec_d_param = dict()
        
        self.Phivec = np.zeros(2 * self.nfreqcomps)
        self.d_Phivec_d_param = dict()
        
        
        self.ptaparams = dict()
        
        self.ptadict = dict()
        
        self.signals = []
        
    
    
    def load_pulsar(self, parfile, timfile, ephem='DE436'):
        psr = Pulsar(parfile, timfile, ephem=ephem)
        return psr
    
    
    def loadSignals(self, incEfac=True, incEquad=True, incRn=True, incOut=True,
                    incTiming=True, incFourier=True, incJitter=False):
        if incEfac:
            self.signals.append(self.add_efac())
        if incEquad:
            self.signals.append(self.add_equad())
        if incRn:
            self.signals.append(self.add_rn())
        if incOut:
            self.signals.append(self.add_outlier())
        if incTiming:
            self.signals.append(self.add_timingmodel())
        if incFourier:
            self.signals.append(self.add_fourier())
        if incJitter:
            pass
        
        index = 0
        for ii, sig in enumerate(self.signals):
            sig['pmin'] = np.array(sig['pmin'])
            sig['pmax'] = np.array(sig['pmax'])
            sig['pstart'] = np.array(sig['pstart'])
            sig['index'] = index
            sig['msk'] = slice(sig['index'], sig['index']+sig['numpars'])
            index += sig['numpars']
        
        for sig in self.signals:
            if sig['type'] == 'rn':
                self.ptaparams.update(dict(zip(sig['name'], sig['pstart'])))
            else:
                self.ptaparams[sig['name']] = sig['pstart'][0]
    
        for ii, key in enumerate(self.ptaparams.keys()):
            if key not in ['timingmodel', 'fouriermode', 'jittermode']:
                self.ptadict[key] = ii
    
    
    def updateParams(self, parameters):
        for key, value in self.ptadict.items():
            self.ptaparams[key] = parameters[value]
    
    
    def add_efac(self):
        efac = parameter.Uniform(0.001, 5.0)
        ef = white_signals.MeasurementNoise(efac=efac, selection=self.selection)
        newsignal = OrderedDict({'type': 'efac',
                                 'name': self.pname + '_efac',
                                 'pmin': [0.001],
                                 'pmax': [5.0],
                                 'pstart': [1.0],
                                 'interval': [True],
                                 'numpars': 1})
        
        self.efac_sig = ef(self.psr)
        return newsignal
    
    
    def add_equad(self):
        equad = parameter.Uniform(-10.0, -4.0)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=self.selection)
        newsignal = OrderedDict({'type': 'equad',
                                 'name': self.pname + '_log10_equad',
                                 'pmin': [-10.0],
                                 'pmax': [-4.0],
                                 'pstart': [-6.5],
                                 'interval': [True],
                                 'numpars': 1})
        self.equad_sig = eq(self.psr)
        return newsignal
    
    
    def add_ecorr(self):
        ecorr = parameter.Uniform(-10.0, -4.0)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=self.selection)
        return ec(self.psr)
    
    
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
        return newsignal
    
    
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
        return newsignal
    
    
    def add_fourier(self):
        npars = 2 * self.nfreqcomps
        newsignal = OrderedDict({'type': 'fouriermode',
                                 'name': 'fouriermode',
                                 'pmin': [-1.0e6]*npars,
                                 'pmax': [1.0e6]*npars,
                                 'pstart': [1.0e-9]*npars,
                                 'interval': [False]*npars,
                                 'numpars': npars})
        
        return newsignal
        
    
    
    def add_outlier(self):
        newsignal = OrderedDict({'type': 'outlier',
                                 'name': self.pname + '_outlierprob',
                                 'pmin': [0.0],
                                 'pmax': [1.0],
                                 'pstart': [0.001],
                                 'interval': [True],
                                 'numpars': 1})
        return newsignal
    
    
    def setWhiteNoise(self, calc_gradient=True):
        self.Nvec[:] = 0
        
        ef = self.efac_sig
        eq = self.equad_sig
        
        self.Nvec[:] = ef.get_ndiag(self.ptaparams) + eq.get_ndiag(self.ptaparams)
        
        if calc_gradient:
            for key, param in self.ptaparams.items():
                if key.endswith('efac'):
                    self.d_Nvec_d_param[self.ptadict[key]] = 2 * self.psr.toaerrs**2 * param
                elif key.endswith('equad'):
                    self.d_Nvec_d_param[self.ptadict[key]] = np.ones_like(self.psr.toas) * \
                                                             2 * np.log(10) * 10**(2*param)
        return
        
        
        
    def setPhi(self, calc_gradient=True):
        self.Phivec[:] = 0
        
        rn = self.rn_sig
        log10A = self.ptaparams[self.pname + '_rn_log10_A']
        gamma = self.ptaparams[self.pname + '_rn_gamma']
        sTmax = self.psr.toas.max() - self.psr.toas.min()
        
        self.Phivec[:] = rn.get_phi(self.ptaparams)
        
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
                self.d_Pb_ind = self.ptadict[key]
        
        return
    
    
    def set_hyperparameters(self, parameters, calc_gradient=True):
        # self.updateParams(parameters)
        
        self.setPhi(calc_gradient=calc_gradient)
        self.setWhiteNoise(calc_gradient=calc_gradient)
        self.setOutliers()
        


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