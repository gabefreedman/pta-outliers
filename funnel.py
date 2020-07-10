#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:20:32 2020

@author: marvin
"""


import numpy as np
import scipy.linalg as sl

from base import ptaLikelihood
from piccard.choleskyext_omp import cython_dL_update_omp


class funnel(ptaLikelihood):
    
    def __init__(self, parfile, timfile):
        super(funnel, self).__init__(parfile, timfile)
        
        self.funnelmin = None
        self.funnelmax = None
        self.funnelstart = None
        
        self.lowLevelStart()
        
    
    def lowLevelStart(self):
        lowlevelpars = ['timingmodel', 'fouriermode', 'jittermode']
        
        if self.basepstart is None:
            return
        
        pstart = self.basepstart.copy()
        
        for key, sig in self.signals.items():
            if sig['type'] in lowlevelpars:
                msk = sig['msk']
                pstart[msk] = 0.1
        
        self.funnelstart = pstart
    
    
    def getLowLevelInds(self):
        slc = np.array([], dtype=np.int)
        
        if 'timingmodel' in self.ptaparams.keys():
            pslc = self.signals['timingmodel']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'fouriermode' in self.ptaparams.keys():
            pslc = self.signals['fouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'dmfouriermode' in self.ptaparams.keys():
            pslc = self.signals['dmfouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'jittermode' in self.ptaparams.keys():
            pslc = self.signals['jittermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        
        return slc
    
    
    def getBeta(self):
        
        Beta_inv_diag = np.zeros(len(self.ZNZ))
        
        if 'fouriermode' in self.ptaparams.keys():
            pslc = self.signals['fouriermode']['msk']
            phivec = self.Phivec
            
            Beta_inv_diag[pslc] = 1.0 / phivec
        
        if 'dmfouriermode' in self.ptaparams.keys():
            pslc = self.signals['dmfouriermode']['msk']
        
        if 'jittermode' in self.ptaparams.keys():
            pslc = self.signals['jittermode']['msk']
        
        return Beta_inv_diag
    
    
    def getSigma(self, Beta_inv_diag):
        Sigma_inv = np.copy(self.ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)
        
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)
        
        L = sl.cholesky(Sigma_inv, lower=True)
        Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)
        cf = (L, True)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv))), L, Li
    
    
    def funnelTransform(self, parameters, set_hyper_params=True, calc_gradient=True):
        
        if set_hyper_params:
            self.set_hyperparameters(parameters, calc_gradient=calc_gradient)
        
        self.fnlslc = self.getLowLevelInds()
        
        self.fnl_Beta_inv = self.getBeta()
        self.fnl_Sigma, self.fnl_L, self.fnl_Li = self.getSigma(self.fnl_Beta_inv)
        self.fnl_mu = np.dot(self.fnl_Sigma, self.ZNyvec)
        
        ii = 0
        log_jacob = 0.0
        gradient = np.zeros_like(parameters)
        
        lowlevel_pars = np.dot(self.fnl_Li.T, parameters[self.fnlslc])
            
            
            


