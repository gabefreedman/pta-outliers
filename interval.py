#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:35:41 2020

@author: marvin
"""


import numpy as np
import scipy.linalg as sl

from funnel import funnel


class interval(funnel):
    
    def __init__(self, parfile, timfile):
        super(interval, self).__init__(parfile, timfile)
        
        self.msk = None
        self.hypMask()
        
        self.initBounds()
        
    
    def initBounds(self):
        self.a = self.funnelmin
        self.b = self.funnelmax
        
    
    def hypMask(self):
        lowlevelpars = ['timingmodel', 'fouriermode', 'jittermode']
        msk = [True] * len(self.basepstart)
        for key, sig in self.signals.items():
            if sig['type'] in lowlevelpars:
                msk[sig['msk']] = [False] * sig['numpars']
        
        self.msk = np.array(msk)
    
    def forward(self, x):
        p = np.atleast_2d(x.copy())
        posinf, neginf = (self.a == x), (self.b == x)
        m = self.msk & ~(posinf | neginf)
        p[:,m] = np.log((p[:,m] - self.a[m]) / (self.b[m] - p[:,m]))
        p[:,posinf] = np.inf
        p[:,neginf] = -np.inf
        return p.reshape(x.shape)


    def backward(self, p):
        x = np.atleast_2d(p.copy())
        m = self.msk
        x[:,m] = (self.b[m] - self.a[m]) * np.exp(x[:,m]) / (1 +
                np.exp(x[:,m])) + self.a[m]
        return x.reshape(p.shape)
    
    
    def dxdp(self, p):
        pp = np.atleast_2d(p)
        m = self.msk
        d = np.ones_like(pp)
        d[:,m] = (self.b[m]-self.a[m])*np.exp(pp[:,m])/(1+np.exp(pp[:,m]))**2
        return d.reshape(p.shape)


    def logjacobian_grad(self, p):
        m = self.msk
        lj = np.sum( np.log(self.b[m]-self.a[m]) + p[m] -
                2*np.log(1.0+np.exp(p[m])) )
    
        lj_grad = np.zeros_like(p)
        lj_grad[m] = (1 - np.exp(p[m])) / (1 + np.exp(p[m]))
        return lj, lj_grad


    def full_loglikelihood_grad(self, parameters):
        
        funnelpars = self.backward(parameters)
        ll, ll_grad = self.funnel_loglikelihood_grad(funnelpars)
        lj, lj_grad = self.logjacobian_grad(parameters)
        
        lp = ll + lj
        lp_grad = ll_grad * self.dxdp(parameters) + lj_grad
        
        return lp, lp_grad


class whitenedLikelihood(interval):
    
    def __init__(self, likob, parameters, hessian):
        self.likob = likob
        self.mu = parameters
        
        self.calc_invsqrt(hessian)
        
    def calc_invsqrt(self, hessian):
        try:
            # Try Cholesky
            self.ch = sl.cholesky(hessian, lower=True)

            # Fast solve
            self.chi = sl.solve_triangular(self.ch, np.eye(len(self.ch)),
                    trans=0, lower=True)
            self.lj = np.sum(np.log(np.diag(self.chi)))
        except sl.LinAlgError:
            # Cholesky fails. Try eigh
            try:
                eigval, eigvec = sl.eigh(hessian)

                if not np.all(eigval > 0):
                    # Try SVD here? Or just regularize?
                    raise sl.LinAlgError("Eigh thinks hessian is not positive definite")
                
                self.ch = eigvec * np.sqrt(eigval)
                self.chi = (eigvec / np.sqrt(eigval)).T
                self.lj = -0.5*np.sum(np.log(eigval))
            except sl.LinAlgError:
                U, s, Vt = sl.svd(hessian)

                if not np.all(s > 0):
                    raise sl.LinAlgError("SVD thinks hessian is not positive definite")

                self.ch = U * np.sqrt(s) # eigvec * np.sqrt(eigval)
                self.chi = (U / np.sqrt(s)).T
                self.lj = -0.5*np.sum(np.log(s))
    
    
    def forward(self, x):
        p = np.atleast_2d(x.copy()) - self.mu
        p = np.dot(self.ch.T, p.T).T
        return p.reshape(x.shape)
    
    
    def backward(self, parameters):
        x = np.atleast_2d(parameters.copy())
        x = np.dot(self.chi.T, x.T).T + self.mu
        return x.reshape(parameters.shape)
    
    
    def logjacobian_grad(self, parameters):
        lj = self.lj
        lj_grad = np.zeros_like(parameters)
        return lj, lj_grad
    
    
    def logposterior_grad(self, parameters):
        x = self.backward(parameters)
        lp, lp_grad = self.likob.full_loglikelihood_grad(x)
        lj, lj_grad = self.logjacobian_grad(parameters)
        grad = np.dot(self.chi, lp_grad) + lj_grad

        return lp + lj, grad