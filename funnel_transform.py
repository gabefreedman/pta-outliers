#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:46:16 2020

@author: marvin

An implementation of non-cenetered reparametrization (Neal's funnel) of the
base likelihood in hubert.py
"""

import numpy as np
import scipy.linalg as sl

from hubert import baseLikelihood
from piccard.choleskyext_omp import cython_dL_update_omp

class funnelLikelihood(baseLikelihood):
    
    def __init__(self, likob):
        super(funnelLikelihood, self).__init__(likob)
    
        self.funnelstart = None
        self.funnelmin = None
        self.funnelmax = None
        self.lowLevelStart()
        self.transformedBounds()
    
    
    def lowLevelStart(self):
        low_level_pars = ['timingmodel_xi', 'fouriermode_xi',
                'dmfouriermode_xi', 'jittermode_xi']

        if self.basepstart is None:
            return

        pstart = self.basepstart.copy()

        for ii, m2signal in enumerate(self.ptasignals):
            if m2signal['stype'] in low_level_pars:
                sind = m2signal['parindex']
                msk = m2signal['bvary']
                npars = np.sum(msk)

                sigstart = np.ones(npars) * 0.1

                pstart[sind:sind+npars] = sigstart

        self.funnelstart = pstart
    
    
    def full_forward(self, x):
        p = np.atleast_2d(x.copy())
        p[0,self.sr_pslc] = np.dot(self.sr_L.T, p[0,self.sr_pslc] - self.sr_mu)
        return p.reshape(x.shape)
    
    
    def full_backward(self, p):
        x = np.atleast_2d(p.copy())
        x[0,self.sr_pslc] = np.dot(self.sr_Li.T, x[0,self.sr_pslc]) + self.sr_mu
        return x.reshape(p.shape)    
    
    
    def transformedBounds(self):
        
        pmin = self.basepmin.copy()
        pmax = self.basepmax.copy()
        
        self.funnel_transformation(pmin)
        self.funnelmin = self.full_forward(pmin)
        
        self.funnel_transformation(pmax)
        self.funnelmax = self.full_forward(pmax)        
    
    
    def get_par_psr_sigma_inds(self):
        slc = np.array([], dtype=np.int)
    
        if self.psr.timingmodelind is not None:
            slc = np.append(slc, np.arange(self.psr.timingmodelind,
                    self.psr.timingmodelind + self.npm))
    
        if self.psr.fourierind is not None:
            slc = np.append(slc, np.arange(self.psr.fourierind,
                    self.psr.fourierind + self.npf))
    
        if self.psr.dmfourierind is not None:
            slc = np.append(slc, np.arange(self.psr.dmfourierind,
                    self.psr.dmfourierind + self.npfdm))
    
        if self.psr.jitterind is not None:
            slc = np.append(slc, np.arange(self.psr.jitterind,
                    self.psr.jitterind + self.npu))
        
        return slc
    
    
    def get_psr_Beta(self):
        
        Beta_inv_diag = np.zeros(len(self.psr.sr_ZNZ))
    
        if self.psr.fourierind is not None:
            # Have a red noise stingray transformation
            findex = np.sum(self.npf[:0])
            nfs = self.npf[0]
            fslc_phi = slice(findex, findex+nfs)
            phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]
    
            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[self.psr.Zmask_F_only] = 1.0 / phivec
    
        if self.psr.dmfourierind is not None:
            # Have a dm noise stingray transformation
            fdmindex = np.sum(self.npfdm[:0])
            nfdms = self.npfdm[0]
            fslc_theta = slice(fdmindex, fdmindex+nfdms)
            thetavec = self.Thetavec[fslc_theta]
    
            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[self.psr.Zmask_D_only] = 1.0 / thetavec
    
        if self.psr.jitterind is not None:
            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[self.psr.Zmask_U_only] = 1.0 / self.Jvec
    
        return Beta_inv_diag
    
    
    def get_psr_Sigma(self, Beta_inv_diag):
        
        Sigma_inv = np.copy(self.psr.sr_ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)
    
        # Construct the full Sigma matrix
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)
    
        # NOTE: Sigma = L_inv^T L_inv    (just easier that way)
        L = sl.cholesky(Sigma_inv, lower=True)
        Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)
        cf = (L, True)
    
        # Turns out that matrix multiplication is faster than cho_solve
        # However, cho_solve is numerically more stable
        return sl.cho_solve(cf, np.eye(len(Sigma_inv))), L, Li
        
        
    
    
    def funnel_transformation(self, parameters, set_hyper_params=True):

        if set_hyper_params:
            self.set_hyperparameters(parameters)
        
        self.sr_pslc = self.get_par_psr_sigma_inds()
    
        # Define the Stingray transformation
        self.sr_Beta_inv = self.get_psr_Beta()
        self.sr_Sigma, self.sr_L, self.sr_Li = self.get_psr_Sigma(self.sr_Beta_inv)
        self.sr_mu = np.dot(self.sr_Sigma, self.psr.sr_ZNyvec)      
        
        ii = 0
        log_jacob = 0.0
        gradient = np.zeros_like(parameters)
    
        # Quantities we need to take derivatives of the Cholesky factor
        lowlevel_pars = np.dot(self.sr_Li.T, parameters[self.sr_pslc])
        self.sr_dL_M, self.sr_dL_tj = cython_dL_update_omp(self.sr_L, self.sr_Li, lowlevel_pars)
    
        # The log(det(Jacobian)), for all low-level parameters
        log_jacob += np.sum(np.log(np.diag(self.sr_Li)))
        
        if self.psr.fourierind is not None:
            fslc_phi = slice(np.sum(self.npf[:ii]),
                    np.sum(self.npf[:ii+1]))
    
            for key, value in self.d_Phivec_d_param.items():
                # Do some slicing magic to get the diagonals of the
                # matrix product in O(n^2) time
                BdB = np.zeros(len(self.sr_Sigma))
                BdB[self.psr.Zmask_F_only] = \
                        self.sr_Beta_inv[self.psr.Zmask_F_only]**2 * \
                        value[fslc_phi]
    
                # We need to remember sr_diagSBS for dxdp_nondiag
                #BS = psr.sr_Sigma * BdB[None, :]
                #psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)
    
                # Log-jacobian for red noise Fourier terms
                gradient[key] += np.sum(
                        self.sr_dL_tj[self.psr.Zmask_F_only] *
                        BdB[self.psr.Zmask_F_only])
    
        # ECORR
        if self.psr.jitterind is not None:
            #fslc_J = slice(np.sum(self.npu[:ii]),
            #        np.sum(self.npu[:ii+1]))
    
            for key, value in self.d_Jvec_d_param.items():
                # Do some slicing magic to get the diagonals of the
                # matrix product in O(n^2) time
                BdB = np.zeros(len(self.sr_Sigma))
                BdB[self.psr.Zmask_U_only] = \
                        self.sr_Beta_inv[self.psr.Zmask_U_only]**2 * \
                        value
    
                # We need to remember sr_diagSBS for dxdp_nondiag
                #BS = psr.sr_Sigma * BdB[None, :]
                #psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)
    
                # Log-jacobian for DM variation Fourier terms
                gradient[key] += np.sum(
                        self.sr_dL_tj[self.psr.Zmask_U_only] *
                        BdB[self.psr.Zmask_U_only])
        
        self.log_jacob = log_jacob
        self.funnel_gradient = gradient
    
    
    def dxdp_nondiag(self, parameters, ll_grad, set_hyper_params=False):
    
        if set_hyper_params:
            self.constructPhi(parameters)
            self.setPsrNoise(parameters)
            self.setPb_outliers(parameters)
            
        if not hasattr(self, "sr_Sigma"):
            raise ValueError("Auxilliary Funnel Transform quantities not set!")
            return
    
        # Quantities we need to take derivatives of the Cholesky factor
        ii=0
        ll_grad2 = np.atleast_2d(ll_grad)
        extra_grad = np.zeros_like(ll_grad2)
        extra_grad[:, :] = np.copy(ll_grad2)
        #pslc_tot = self.get_par_psr_sigma_inds(ii, psr)
        pslc_tot = self.sr_pslc
        ll_grad2_psr = ll_grad2[:, pslc_tot]
    
        # We have to do the 'regular' dxdp here as well, since in this full
        # Stingray transform, that is a full 2D matrix for the low-level
        # parameters
        extra_grad[:, pslc_tot] = np.dot(self.sr_Li, ll_grad2_psr.T).T
    
        if self.psr.fourierind is not None:
            fslc_phi = slice(np.sum(self.npf[:ii]), np.sum(self.npf[:ii+1]))
            slc_sig = self.psr.Zmask_F_only
    
            for key, d_Phivec_d_p in self.d_Phivec_d_param.items():
                BdB = np.zeros(len(self.sr_Sigma))
                BdB[self.psr.Zmask_F_only] = \
                        self.sr_Beta_inv[self.psr.Zmask_F_only]**2 * \
                        d_Phivec_d_p[fslc_phi]
    
                # dxdp for Sigma
                dxdhp = np.dot(self.sr_Li.T, np.dot(self.sr_dL_M[:,slc_sig],
                        BdB[self.psr.Zmask_F_only]))
                extra_grad[:, key] += np.sum(
                        dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)
    
                # dxdp for mu
                WBWv = np.dot(self.sr_Sigma[:,slc_sig],
                        self.sr_Beta_inv[slc_sig]**2 *
                        d_Phivec_d_p[fslc_phi] * self.sr_mu[slc_sig])
                extra_grad[:, key] += np.sum(ll_grad2_psr * 
                        WBWv[None, :], axis=1)
    
        if self.psr.jitterind is not None:
            slc_sig = self.psr.Zmask_U_only
    
            for key, d_Jvec_d_p in self.d_Jvec_d_param.items():
                BdB = np.zeros(len(self.sr_Sigma))
                BdB[self.psr.Zmask_U_only] = \
                        self.sr_Beta_inv[self.psr.Zmask_U_only]**2 * \
                        d_Jvec_d_p
                # dxdp for Sigma
                dxdhp = np.dot(self.sr_Li.T, np.dot(self.sr_dL_M[:,slc_sig],
                        BdB[self.psr.Zmask_U_only]))
                extra_grad[:, key] += np.sum(
                        dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)
    
                # dxdp for mu
                WBWv = np.dot(self.sr_Sigma[:,slc_sig],
                        self.sr_Beta_inv[slc_sig]**2 *
                        d_Jvec_d_p * self.sr_mu[slc_sig])
                extra_grad[:, key] += np.sum(ll_grad2_psr * 
                        WBWv[None, :], axis=1)
    
    
        return extra_grad.reshape(ll_grad.shape)
    
    
    def funnel_loglikelihood_grad(self, parameters):
        
        self.funnel_transformation(parameters)
        basepars = self.full_backward(parameters)
        
        ll, ll_grad = self.base_loglikelihood_grad(basepars)
        lj, lj_grad = self.log_jacob, self.funnel_gradient
        
        lp = ll + lj
        lp_grad = lj_grad + self.dxdp_nondiag(parameters, ll_grad)
        
        return lp, lp_grad

