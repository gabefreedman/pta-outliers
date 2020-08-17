#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:09:02 2020

@author: marvin
"""


import numpy as np
import scipy.linalg as sl

from jitterext import cython_Uj
from pulsar import HierarchicalPTA
import utils as ut


class ptaLikelihood(HierarchicalPTA):

    def __init__(self, parfile, timfile):
        super(ptaLikelihood, self).__init__(parfile, timfile)

        self.basepmin = None
        self.basepmax = None
        self.basepstart = None

        self.Mmat_g = self.gibbs_set_design()
        self.Zmat = None
        self.ZNZ = None
        self.ZNyvec = None

        self.outlier_prob = None
        self.detresiduals = None
        self.outlier_sig_dict = dict()
        self.d_Pb_ind = None

        self.d_L_d_b = None
        self.d_Pr_d_b = None


        self.setBounds()
        self.setZmat()
        self.setFunnelAuxiliary()


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
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_F], full_matrices=False)
            Mmat_g[:, Mmask_F] = U

        if np.sum(Mmask_D) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_D], full_matrices=False)
            Mmat_g[:, Mmask_D] = U

        if np.sum(Mmask_U) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_U], full_matrices=False)
            Mmat_g[:, Mmask_U] = U

        if np.sum(Mmask_M) > 0:
            U, _, _ = sl.svd(self.psr.Mmat[:, Mmask_M], full_matrices=False)
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
        for _, sig in self.signals.items():
            pmin.extend(sig['pmin'])
            pmax.extend(sig['pmax'])
            pstart.extend(sig['pstart'])

        self.basepmin = np.array(pmin)
        self.basepmax = np.array(pmax)
        self.basepstart = np.array(pstart)


    def updateParams(self, parameters):
        for key, value in self.ptadict.items():
            self.ptaparams[key] = parameters[value]


    def setWhiteNoise(self, calc_gradient=True):
        self.Nvec[:] = 0
        self.Jvec[:] = 0

        ef = self.efac_sig
        eq = self.equad_sig
        ec = self.ecorr_sig

        self.Nvec[:] = ef.get_ndiag(self.ptaparams) + eq.get_ndiag(self.ptaparams)

        if ec:
            for param in ec.param_names:
                pequadsqr = 10**(2*self.ptaparams[param])
                self.Jvec += self.signals[param]['Jvec'] * pequadsqr


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


    def setPhi(self, calc_gradient=True):
        self.Phivec[:] = 0

        rn = self.rn_sig
        log10A = self.ptaparams[self.pname + '_rn_log10_A']
        gamma = self.ptaparams[self.pname + '_rn_gamma']
        sTmax = self.psr.toas.max() - self.psr.toas.min()

        self.Phivec[:] = rn.get_phi(self.ptaparams)

        if calc_gradient:
            d_mat = ut.d_powerlaw(log10A, gamma, sTmax, self.Ffreqs)
            for key, _ in self.ptaparams.items():
                if key.endswith('log10_A'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:, 0]
                elif key.endswith('gamma'):
                    self.d_Phivec_d_param[self.ptadict[key]] = d_mat[:, 1]


    def setOutliers(self):
        for key, param in self.ptaparams.items():
            if key.endswith('outlierprob'):
                self.outlier_prob = param
                self.d_Pb_ind = [self.ptadict[key]]


    def setDetSources(self, parameters, calc_gradient=True):
        d_L_d_b = np.zeros_like(parameters)
        d_Pr_d_b = np.zeros_like(parameters)
        self.outlier_sig_dict = dict()

        self.detresiduals = self.psr.residuals.copy()

        for _, sig in self.signals.items():
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
            for _, sig in self.signals.items():
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
            gradient[pslc] += np.sum(d_L_d_b_o * bigL0[None, :]/bigL[None, :], axis=1)

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
