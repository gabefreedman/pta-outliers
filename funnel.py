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
        
        self.Zmask_M = None
        self.Zmask_F = None
        self.Zmask_U = None
        self.getLowLevelZmask()
        
        self.lowLevelStart()
        self.transformedBounds()
        
    
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
    
    
    def full_forward(self, x):
        p = np.atleast_2d(x.copy())
        p[0,self.fnlslc] = np.dot(self.fnl_L.T, p[0,self.fnlslc] - self.fnl_mu)
        return p.reshape(x.shape)
    
    
    def full_backward(self, p):
        x = np.atleast_2d(p.copy())
        x[0,self.fnlslc] = np.dot(self.fnl_Li.T, x[0,self.fnlslc]) + self.fnl_mu
        return x.reshape(p.shape)    
    
    
    def multi_full_backward(self, p):
        # Hacky way to fix when I backward transform all 20000+ samples at once
        # Note this will ONLY work for 2d arrays with more than one column
        x = np.atleast_2d(p.copy())
        for ii, xx in enumerate(x):
            self.funnelTransform(xx)
            x[ii,self.fnlslc] = np.dot(self.fnl_Li.T, x[ii,self.fnlslc]) + self.fnl_mu
        return x.reshape(p.shape)
            
    
    
    def transformedBounds(self):
        
        pmin = self.basepmin.copy()
        pmax = self.basepmax.copy()
        
        self.funnelTransform(pmin)
        self.funnelmin = self.full_forward(pmin)
        
        self.funnelTransform(pmax)
        self.funnelmax = self.full_forward(pmax)
    
    
    def getLowLevelMask(self):
        slc = np.array([], dtype=np.int)
        
        if 'timingmodel' in self.signals.keys():
            pslc = self.signals['timingmodel']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'fouriermode' in self.signals.keys():
            pslc = self.signals['fouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'dmfouriermode' in self.signals.keys():
            pslc = self.signals['dmfouriermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        if 'jittermode' in self.signals.keys():
            pslc = self.signals['jittermode']['msk']
            slc = np.append(slc, np.arange(pslc.start, pslc.stop))
        
        return slc
    
    
    def getLowLevelZmask(self):
        slc = np.array([], dtype=np.int)
        
        if 'timingmodel' in self.signals.keys():
            npars = self.signals['timingmodel']['numpars']
            self.Zmask_M = slice(0, npars)
            slc = np.append(slc, np.arange(0, npars))
        if 'fouriermode' in self.signals.keys():
            npars = self.signals['fouriermode']['numpars']
            if slc.size > 0:
                self.Zmask_F = slice(slc[-1]+1, slc[-1]+1+npars)
                slc = np.append(slc, np.arange(slc[-1]+1, slc[-1]+1+npars))
            else:
                self.Zmask_F = slice(0, npars)
                slc = np.append(slc, np.arange(0, npars))
        if 'jittermode' in self.signals.keys():
            npars = self.signals['jittermode']['numpars']
            if slc.size:
                self.Zmask_U = slice(slc[-1]+1, slc[-1]+1+npars)
                slc = np.append(slc, np.arange(slc[-1]+1, slc[-1]+1+npars))
            else:
                self.Zmask_U = slice(0, npars)
                slc = np.append(slc, np.arange(0, npars))
        
        return slc
        
        
    
    def getBeta(self):
        
        Beta_inv_diag = np.zeros(len(self.ZNZ))
        
        if 'fouriermode' in self.ptaparams.keys():
            phivec = self.Phivec
            
            Beta_inv_diag[self.Zmask_F] = 1.0 / phivec
        
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
        
        self.fnlslc = self.getLowLevelMask()
        
        self.fnl_Beta_inv = self.getBeta()
        self.fnl_Sigma, self.fnl_L, self.fnl_Li = self.getSigma(self.fnl_Beta_inv)
        self.fnl_mu = np.dot(self.fnl_Sigma, self.ZNyvec)
        
        log_jacob = 0.0
        gradient = np.zeros_like(parameters)
        
        lowlevel_pars = np.dot(self.fnl_Li.T, parameters[self.fnlslc])
        self.fnl_dL_M, self.fnl_dL_tj = cython_dL_update_omp(self.fnl_L, self.fnl_Li, lowlevel_pars)
        
        log_jacob += np.sum(np.log(np.diag(self.fnl_Li)))
        
        if 'fouriermode' in self.ptaparams.keys():
            
            for key, d_Phivec_d_param in self.d_Phivec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_F] = self.fnl_Beta_inv[self.Zmask_F]**2 * d_Phivec_d_param
                
                gradient[key] += np.sum(self.fnl_dL_tj[self.Zmask_F] * BdB[self.Zmask_F])
                
        if 'jittermode' in self.ptaparams.keys():
            
            for key, d_Jvec_d_param in self.d_Jvec_d_param.items():
                BdB = np.zeros(len(self.fnl_Sigma))
                BdB[self.Zmask_U] = self.fnl_Beta_inv[self.Zmask_U]**2 * d_Jvec_d_param
                
                gradient[key] += np.sum(self.fnl_dL_tj[self.Zmask_U] * BdB[self.Zmask_U])
        
        self.log_jacob = log_jacob
        self.funnel_gradient = gradient
        
            
            


