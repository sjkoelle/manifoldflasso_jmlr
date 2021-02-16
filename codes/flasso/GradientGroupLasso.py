from einops import rearrange

import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad
from autograd import grad

import logging
from copy import deepcopy

#import numpy as np
from scipy.special import expit
from pyglmnet import utils


class GradientGroupLasso:
    
    def __init__(self, dg_M, df_M, reg_l1s, reg_l2, max_iter,learning_rate, tol, beta0_npm= None):
        
        n = dg_M.shape[0]
        d= dg_M.shape[1]
        m = df_M.shape[2]
        p = dg_M.shape[2]
        dummy_beta = np.ones((n,p,m))
        
        self.dg_M = dg_M
        self.df_M = df_M
        self.reg_l1s = reg_l1s
        self.reg_l2 = reg_l2
        self.beta0_npm = beta0_npm
        self.n = n
        self.p = p
        self.m = m 
        self.d = d
        self.dummy_beta = dummy_beta
        #self.group = np.asarray(group)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.Tau = None
        self.alpha = 1.
        self.lossresults = {}
        self.dls = {}
        self.l2loss = {}
        self.penalty = {}
        
    def _prox(self,beta_npm, thresh):
        """Proximal operator."""
        
        p = self.p
        result = np.zeros(beta_npm.shape)
        result = np.asarray(result, dtype = float)
        for j in range(p):
            if np.linalg.norm(beta_npm[:,j,:]) > 0.:
                potentialoutput = beta_npm[:,j,:] - (thresh / np.linalg.norm(beta_npm[:,j,:])) * beta_npm[:,j,:]
                posind = np.asarray(np.where(beta_npm[:,j,:] > 0.))
                negind = np.asarray(np.where(beta_npm[:,j,:] < 0.))
                po = beta_npm[:,j,:].copy()
                po[posind[0],posind[1]] = np.asarray(np.clip(potentialoutput[posind[0],posind[1]],a_min = 0., a_max = 1e15), dtype = float)
                po[negind[0],negind[1]] = np.asarray(np.clip(potentialoutput[negind[0],negind[1]],a_min = -1e15, a_max = 0.), dtype = float)
                result[:,j,:] = po
        return result

    def _grad_L2loss(self, beta_npm):
        
        df_M = self.df_M
        dg_M = self.dg_M
        reg_l2 = self.reg_l2
        dummy_beta = self.dummy_beta
        
        df_M_hat = np.einsum('ndp,npm->ndm',dg_M, beta_npm)
        error = df_M_hat - df_M
        grad_beta = np.einsum('ndm,ndp->npm',error,dg_M) #+ reg_l2 * np.ones()
        #if 
        return grad_beta
    
    def _L1penalty(self, beta_npm):
        
        p = self.p
        m = self.m
        n = self.n 
        beta_mn_p = rearrange(beta_npm, 'n p m -> (m n) p')#np.reshape(beta_mnp, ((m*n,p)))
        L1penalty = np.linalg.norm(beta_mn_p, axis = 0).sum()
        
        return L1penalty
    
    def _loss(self,beta_npm, reg_lambda):
        """Define the objective function for elastic net."""
        L = self._logL(beta_npm)
        P = self._L1penalty(beta_npm)
        J = -L + reg_lambda * P
        return J
    
    def _logL(self,beta_npm):
        
        df_M = self.df_M
        dg_M = self.dg_M
        
        df_M_hat = np.einsum('ndp,npm -> ndm',dg_M, beta_npm)
        logL = -0.5 * np.linalg.norm((df_M - df_M_hat))**2
        return(logL)
    
    def _L2loss(self,beta_npm):
        output = -self._logL(beta_npm)
        return(output)

    def fhatlambda(self,learning_rate,beta_npm_new,beta_npm_old):

        #print('lr',learning_rate)
        output = self._L2loss(beta_npm_old) + np.einsum('npm,npm', self._grad_L2loss(beta_npm_old),(beta_npm_new-beta_npm_old)) + (1/(2*learning_rate)) * np.linalg.norm(beta_npm_new-beta_npm_old)**2
        
        return(output)

    def _btalgorithm(self,beta_npm ,learning_rate,b,maxiter_bt,rl):
        
        grad_beta = self._grad_L2loss(beta_npm = beta_npm)
        for i in range(maxiter_bt):
            beta_npm_postgrad = beta_npm - learning_rate * grad_beta
            beta_npm_postgrad_postprox = self._prox(beta_npm_postgrad, learning_rate * rl)
            fz = self._L2loss(beta_npm_postgrad_postprox)
            #fhatz = self.fhatlambda(lam,beta_npm_postgrad_postprox, beta_npm_postgrad)
            fhatz = self.fhatlambda(learning_rate,beta_npm_postgrad_postprox, beta_npm)
            if fz <= fhatz:
                #print(i)
                break
            learning_rate = b*learning_rate    
            
        return(beta_npm_postgrad_postprox,learning_rate)
    
    def fit(self, beta0_npm = None):

        reg_l1s = self.reg_l1s
        n = self.n
        m = self.m
        p = self.p
        
        dg_M = self.dg_M
        df_M = self.df_M
        
        tol = self.tol
        np.random.RandomState(0)
        
        if beta0_npm is None:
            beta_npm_hat = 1 / (n*m*p) * np.random.normal(0.0, 1.0, [n, p,m])
            #1 / (n_features) * np.random.normal(0.0, 1.0, [n_features, n_classes])
        else: 
            beta_npm_hat = beta0_npm
            
        fit_params = list()
        for l, rl in enumerate(reg_l1s):
            fit_params.append({'beta': beta_npm_hat})
            if l == 0:
                fit_params[-1]['beta'] = beta_npm_hat
            else:
                fit_params[-1]['beta'] = fit_params[-2]['beta']
            
            alpha = 1.
            beta_npm_hat = fit_params[-1]['beta']
            #g = np.zeros([n_features, n_classes])
            L, DL ,L2,PEN = list(), list() , list(), list()
            learning_rate = self.learning_rate
            beta_npm_hat_1 = beta_npm_hat.copy()
            beta_npm_hat_2 = beta_npm_hat.copy()
            for t in range(0, self.max_iter):
                #print(t,l,rl)
                #print(t)
                L.append(self._loss(beta_npm_hat, rl))
                L2.append(self._L2loss(beta_npm_hat))
                PEN.append(self._L1penalty(beta_npm_hat))
                w = (t / (t+ 3))
                beta_npm_hat_momentumguess = beta_npm_hat + w*(beta_npm_hat_1 - beta_npm_hat_2)
                
                beta_npm_hat , learning_rate = self._btalgorithm(beta_npm_hat_momentumguess,learning_rate,.5,1000, rl)
                #print(beta_npm_hat_momentumguess.max(), beta_npm_hat.max(),self._L2loss(beta_npm_hat), learning_rate)
                beta_npm_hat_2 = beta_npm_hat_1.copy()
                beta_npm_hat_1 = beta_npm_hat.copy()
                
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < tol:
                        print('converged', rl)
                        msg = ('\tConverged. Loss function:'
                               ' {0:.2f}').format(L[-1])
                        msg = ('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                        break

            fit_params[-1]['beta'] = beta_npm_hat
            self.lossresults[rl] = L
            self.l2loss[rl] = L2
            self.penalty[rl] = PEN
            self.dls[rl] = DL

        self.fit_ = fit_params
        #self.ynull_ = np.mean(y)

        return self


def get_sr_lambda_sam_parallel(replicate, gl_itermax, lambdas_start,reg_l2, max_search, card, tol,learning_rate):
    
    print('initializing lambda search')
    dg_M = replicate.dg_M
    df_M =  replicate.df_M
    highprobes = np.asarray([])
    lowprobes = np.asarray([])
    #lambdas_start = np.asarray([0,1])
    probe_init_low = lambdas_start[0]
    probe_init_high = lambdas_start[1]

    coeffs = {}
    combined_norms = {}

    GGL = GradientGroupLasso(dg_M, df_M, np.asarray([probe_init_low]), reg_l2, gl_itermax,learning_rate, tol, beta0_npm= None)
    GGL.fit()
    beta0_npm = GGL.fit_[-1]['beta']
    coeffs[probe_init_low] = GGL.fit_[-1]['beta']
    combined_norms[probe_init_low] = np.sqrt((np.linalg.norm(coeffs[probe_init_low], axis = 0)**2).sum(axis = 1))

    GGL = GradientGroupLasso(dg_M, df_M, np.asarray([probe_init_high]), reg_l2, gl_itermax,learning_rate, tol, beta0_npm= beta0_npm)
    GGL.fit(beta0_npm = beta0_npm)
    #beta0_npm = GGL.fit_[-1]['beta']
    coeffs[probe_init_high] = GGL.fit_[-1]['beta']
    combined_norms[probe_init_high] = np.sqrt((np.linalg.norm(coeffs[probe_init_high], axis = 0)**2).sum(axis = 1))

    n_comp = len(np.where(~np.isclose(combined_norms[probe_init_high],0.,1e-12))[0])
    #n_comp = len(np.where(combined_norms[probe_init_high] != 0)[0])
    lowprobes = np.append(lowprobes, probe_init_low)

    if n_comp == card:
        #high_int = probe
        print('we did it',np.where(~np.isclose(combined_norms[probe_init_high],0.,1e-12))[0])
        return (probe_init_high, coeffs, combined_norms)

    if n_comp < card:
        highprobes = np.append(highprobes, probe_init_high)
        probe = (lowprobes.max() + highprobes.min()) / 2
    if n_comp > card:
        lowprobes = np.append(lowprobes, probe_init_high)
        probe = lowprobes.max() * 2

    for i in range(max_search):
        print(i, probe, 'probe')
        beta0_npm = coeffs[lowprobes.max()]
        if not np.isin(probe, list(combined_norms.keys())):
            #print('probe',probe)
            GGL = GradientGroupLasso(dg_M, df_M, np.asarray([probe]), reg_l2, gl_itermax,learning_rate, tol, beta0_npm = beta0_npm)
            GGL.fit(beta0_npm = beta0_npm)
            coeffs[probe] = GGL.fit_[-1]['beta']
            combined_norms[probe] = np.sqrt((np.linalg.norm(coeffs[probe], axis = 0)**2).sum(axis = 1))

        #np.where(~np.isclose(np.linalg.norm(np.linalg.norm(GGL.fit_[-1]['beta'], axis=2), axis=0) ,0., 1e-16))[0]
        #n_comp = len(np.where(combined_norms[probe] != 0)[0])
        n_comp = len(np.where(~np.isclose(combined_norms[probe],0.,1e-12))[0])
        if n_comp == card:
            #high_int = probe
            print('we did it',np.where(~np.isclose(combined_norms[probe],0.,1e-12))[0])
            return(probe, coeffs, combined_norms)
        else:
            if n_comp < card:
                highprobes = np.append(highprobes, probe)
            if n_comp > card:
                lowprobes = np.append(lowprobes, probe)
            if len(highprobes) > 0:
                probe = (lowprobes.max() + highprobes.min()) / 2
            else:
                probe = lowprobes.max() * 2
                
def batch_stream(replicates):
    
    reps = np.asarray(list(replicates.keys()))
    nreps= len(reps)
    for r in range(nreps):
        yield(replicates[r])
    