__author__ = 'qfeng'

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats
import random

from scipy.stats import chisquare
from scipy.special import factorial, gamma
from scipy.stats import poisson, chi2
from scipy.stats import norm

def gaus(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def poisson_pdf(x, u):
    """
    The probability of observing x events given the expected number of events is u
    """
    #return np.exp(-u)*(u**x)/factorial(x)
    #return np.exp(-u)*(u**x)/gamma(x+1)
    return poisson.pmf(x, u)

def counting_pdf(x, y, u, b, tau=1):
    """
    Given the expected signal u and expected bkg b,
    the Poisson probability of observing x ON events and y OFF events.
    Note alpha is 1 here.
    """
    return poisson_pdf(x, u+b)*poisson_pdf(y, tau*b)

def likeihood_ratio(u, x, y, tau=1):
    # Rolke's UL assuming simple Poisson for on (X) and off (Y) with eff=1.
    #maximizing likelihood function over both u and b, we have
    #MLE u_hat=x-y/tau, b_hat=y/tau, where x is the observed ON counts, and y OFF, and tau=1./alpha;
    #while given u, maximizing likelihood function over b, we have
    #b_hat(u) = (x+y-(1.+tau)*u+np.sqrt((x+y-(1.+tau)*u)**2+4.*(1.+tau)*y*u))/2./(1.+tau)
    b_hat_u = (x+y-(1.+tau)*u+np.sqrt((x+y-(1.+tau)*u)**2+4.*(1.+tau)*y*u))/2./(1.+tau)
    return counting_pdf(x, y, u, b_hat_u, tau=tau)*1.0/counting_pdf(x, y, x-y, y, tau=tau)


class UL_on_off(object):
    """
    UL class from known ON, OFF
    """
    def __init__(self, x, y, tau=1.0):
        self.ul=0.0
        #y is measured number of OFF events in bkg region
        self.y=x
        #x is measured number of ON events in the signal region
        self.x=y
        #tau is 1./alpha
        self.tau=tau

    def set_us(self, us):
        #us is the range of expected signal to search for likelihood
        self.us = us
    def set_bs(self, bs):
        #us is the range of expected bkg to search for likelihood
        self.bs = bs
    def make_mesh(self, us=None, bs=None):
        if us is not None:
            self.set_us(us)
        if bs is not None:
            self.set_bs(bs)
        if hasattr(self, 'us') and hasattr(self, 'bs'):
            self.X, self.Y = np.meshgrid(self.us, self.bs, indexing='ij')
            self.Z = np.zeros((self.us.shape[0], self.bs.shape[0]))
            for i, u in enumerate(self.us):
                for j, b in enumerate(self.bs):
                    self.Z[i,j] = counting_pdf(self.x, self.y, u, b, self.tau)
        else:
            print("Set us and bs first!")

    def search_mesh(self, us=None, bs=None, p0=None):
        self.make_mesh(us=us, bs=bs)
        self.get_marginal()
        if p0 is not None:
            self.fit_gaus(p0=p0)

    def get_marginal(self):
        self.cdfs = np.sum(self.Z, axis=1)
        self.cdfs_off = np.sum(self.Z, axis=0)

    def fit_gaus(self, p0 = [1.e-2, 0., 1.], bkg=False):
        if not hasattr(self, 'cdfs'):
            self.get_marginal()
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        if not bkg:
            popt, pcov = curve_fit(gaus, self.us, self.cdfs, p0=p0)
            self.fit_A, self.fit_mu, self.fit_sigma = popt
            self.fit_dA, self.fit_dmu, self.fit_dsigma = np.sqrt(np.diag(pcov))
            return self.fit_A, self.fit_mu, self.fit_sigma
        else:
            popt, pcov = curve_fit(gaus, us, cdfs, p0=p0)
            self.fit_A_off, self.fit_mu_off, self.fit_sigma_off = popt
            self.fit_dA_off, self.fit_dmu_off, self.fit_dsigma_off = np.sqrt(np.diag(pcov))
            return self.fit_A_off, self.fit_mu_off, self.fit_sigma_off

    def plot_pdf(self):
        plt.subplot(1, 1, 1)
        im = plt.pcolormesh(X, Y, Z, cmap=cm.coolwarm)
        plt.colorbar()

    def get_DH_UL(self, cl=0.9):
        if not hasattr(self, 'fit_mu'):
            print("Need to fit first! Doing it...")
            self.fit_gaus()
        self.norm_negative = norm.cdf(0, loc=self.fit_mu, scale=self.fit_sigma)
        self.norm_positive = 1. - self.norm_negative
        self.ul = norm.ppf(self.norm_negative + cl*self.norm_positive, loc=self.fit_mu, scale=self.fit_sigma)
        print("David Hanna's %d%% UL is %.2f" % (cl*100, self.ul))
        return self.ul

    def get_Rolke_UL(self, cl=0.9, tol=5e-3):
        ll_thresh = chi2.ppf(cl, 1)
        lls = []
        for u in self.us:
            ll_ = -2.*np.log(likeihood_ratio(u, self.x, self.y))
            lls.append(ll_)
        lls = np.array(lls)
        slice_ = np.where(abs(lls-ll_thresh)<tol)
        if lls[slice_].shape[0]==2:
            print("Rolke's %d%% confidence range: %.2f - %.2f" % (cl*100, lls[slice_][0], lls[slice_][1]))
            return lls[slice_]
        elif lls[slice_].shape[0]==1:
            print("Found one Rolke's %d%% confidence limit: %.2f" % (cl*100, lls[slice_]))
            return lls[slice_]
        elif lls[slice_].shape[0]>2:
            tol /= 10.
            self.get_Rolke_UL(cl=cl, tol=tol)
        elif lls[slice_].shape[0]==0:
            tol *= 5.
            self.get_Rolke_UL(cl=cl, tol=tol)

def test():
    ul1 = UL_on_off(44710., 44302.)
    ul1.search_mesh(us=np.arange(-500, 1300), bs=np.arange(43500, 45000), p0 = [1.e-3, 408, 300.])
    print ul1.get_DH_UL()
    print ul1.get_Rolke_UL()
    return test()

if __name__=='__main__':
    test()