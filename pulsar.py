#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:22:43 2020

@author: marvin
"""

import numpy as np
from collections import OrderedDict

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import selections, white_signals, gp_signals, utils
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise.signals.utils import create_quantization_matrix, quant2ind

import libstempo as lt

import utils as ut


class HierarchicalPTA(object):
    
    def __init__(self, parfile, timfile, 
                 selection=selections.Selection(selections.by_backend),
                 nfreqcomps=30):
        self.parfile = parfile
        self.timfile = timfile
        self.selection = selection
        
        # Initialize libstempo and enterprise pulsar objects
        self.ltpsr = None
        self.psr = None
        
        self.pname = None
        
        # Initialize white noise vectors and gradients
        self.Nvec = None
        self.Jvec = None
        
        self.d_Nvec_d_param = dict()
        self.d_Jvec_d_param = dict()
        
        # Initialize red noise and jitter mode vectors and gradients
        self.Phivec = None
        
        self.d_Phivec_d_param = dict()
        
        # Initialize Fourier design matrix
        self.nfreqcomps = nfreqcomps
        self.Fmat = None
        self.Ffreqs = None
        
        # Initialize ECORR exploder matrix
        self.Umat = None
        self.Uindslc = None
        
        # Enterprise signals
        self.efac_sig = None
        self.equad_sig = None
        self.ecorr_sig = None
        self.rn_sig = None
        self.tm_sig = None
        
        # Signal and parameter dictionaries
        self.signals = dict()
        self.ptadict = dict()
        self.ptaparams = dict()
        
        self.init_hierarchical_model(parfile, timfile)
    
    def init_hierarchical_model(self, parfile, timfile):
        
        self.load_t2pulsar(parfile, timfile)        
        self.psr = self.load_entpulsar(parfile, timfile)
        self.ltpsr = None
        
        self.set_Fmat_auxiliaries()
        self.set_Umat_auxiliaries()
        
        self.Nvec = np.zeros(len(self.psr.toas))
        self.Jvec = np.zeros(self.Umat.shape[1])
        self.Phivec = np.zeros(2 * self.nfreqcomps)
        
        self.loadSignals()
    
    
    def set_Umat_auxiliaries(self):
        Umat, _ = create_quantization_matrix(self.psr.toas)
        Uind = quant2ind(Umat)
        
        self.Umat, self.Uindslc = ut.set_Uindslc(Umat, Uind)
    
    
    def set_Fmat_auxiliaries(self):
        Fmat, self.Ffreqs = createfourierdesignmatrix_red(self.psr.toas)
        self.Fmat = np.zeros_like(Fmat)
        self.Fmat[:, 1::2] = Fmat[:, ::2]
        self.Fmat[:, ::2] = Fmat[:, 1::2]
    
    
    def load_t2pulsar(self, parfile, timfile):
        self.ltpsr = lt.tempopulsar(parfile, timfile)
        self.ephem = self.ltpsr.ephemeris
        self.F0 = self.ltpsr['F0'].val
        self.P0 = 1.0 / self.F0
    
    
    def load_entpulsar(self, parfile, timfile):
        toas = np.double(np.array(self.ltpsr.toas())-53000.0)*86400
        residuals = np.double(self.ltpsr.residuals())
        toaerrs = np.double(1e-6*self.ltpsr.toaerrs)

        if 'f' in self.ltpsr.flags():
            flags = self.ltpsr.flagvals('f')
        else:
            flags = self.ltpsr.flagvals('be')

        isort, _ = ut.argsortTOAs(toas, flags)

        psr = Pulsar(parfile, timfile, ephem=self.ephem, sort=False)
        self.pname = psr.name
        
        psr._isort = isort

        psr._toas = toas
        psr._residuals = residuals
        psr._toaerrs = toaerrs

        return psr
    
    
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
