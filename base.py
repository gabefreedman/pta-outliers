#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:09:02 2020

@author: marvin
"""


import numpy as np

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
        
        self.Nvec = np.zeros(len(self.psr.toas))
        self.d_Nvec_d_param = dict()
        
        self.Phivec = np.zeros(2 * self.nfreqcomps)
        self.d_Phivec_d_param = dict()
        
        
        self.ptaparams = {self.pname + '_efac': 1.0,
                          self.pname + '_log10_equad': -6.5,
                          self.pname + '_rn_log10_A': -14.5,
                          self.pname + '_rn_gamma': 3.51}
        
        self.ptadict = {self.pname + '_efac': 0,
                        self.pname + '_log10_equad': 1,
                        self.pname + '_rn_log10_A': 2,
                        self.pname + '_rn_gamma': 3}
    
    
    def load_pulsar(self, parfile, timfile, ephem='DE436'):
        psr = Pulsar(parfile, timfile, ephem=ephem)
        return psr
    
    
    def updateParams(self, parameters):
        for key, value in self.ptadict.items():
            self.ptaparams[key] = parameters[value]
    
    
    def add_efac(self):
        efac = parameter.Uniform(0.001, 5.0)
        ef = white_signals.MeasurementNoise(efac=efac, selection=self.selection)
        return ef(self.psr)
    
    
    def add_equad(self):
        equad = parameter.Uniform(-10.0, -4.0)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=self.selection)
        return eq(self.psr)
    
    
    def add_ecorr(self):
        ecorr = parameter.Uniform(-10.0, -4.0)
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=self.selection)
        return ec(self.psr)
    
    
    def add_rn(self):
        log10_A = parameter.Uniform(-20.0, -10.0)
        gamma = parameter.Uniform(0.02, 6.98)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(pl, components=self.nfreqcomps, name='rn')
        return rn(self.psr)
    
    
    def add_timingmodel(self):
        tm = gp_signals.TimingModel(use_svd=False)
        return tm(self.psr)
    
    
    def setWhiteNoise(self, calc_gradient=True):
        self.Nvec[:] = 0
        
        ef = self.add_efac()
        eq = self.add_equad()
        
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
        
        rn = self.add_rn()
        log10A = self.ptaparams[self.pname + '_rn_log10_A']
        gamma = self.ptaparams[self.pname + '_rn_gamma']
        sTmax = self.psr.toas.max() - self.psr.toas.min()
        _, Ffreqs = createfourierdesignmatrix_red(self.psr.toas)
        
        self.Phivec[:] = rn.get_phi(self.ptaparams)
        
        if calc_gradient:
            d_mat = d_powerlaw(log10A, gamma, sTmax, Ffreqs)
            for key, param in self.ptaparams.items():
                if key.endswith('log10_A'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:,0]
                elif key.endswith('gamma'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:,1]
        
        return
    
    
    def set_hyperparameters(self, parameters, calc_gradient=True):
        self.updateParams(parameters)
        
        self.setPhi(calc_gradient=calc_gradient)
        self.setWhiteNoise(calc_gradient=calc_gradient)
        


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