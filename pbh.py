__author__ = 'qfeng'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import os
import pandas as pd
from scipy.optimize import curve_fit, minimize
#from scipy.optimize import minimize

#from scipy import stats
import random
import shutil
from scipy import integrate


import sys
if (sys.version_info > (3, 0)):
     # Python 3 code in this block
     import _pickle as pickle
else:
     # Python 2 code in this block
     import cPickle as pickle
from scipy.special import gamma
from math import factorial
import tables
from optparse import OptionParser

import socket

sys.setrecursionlimit(50000)

try:
    import ROOT
    ROOT.PyConfig.StartGuiThread = False
except:
    print("Can't import ROOT, no related functionality possible")

#import time

#try:
#    from numba import jit, autojit
#except:
#    print("Numba not installed")

def deg2rad(deg):
    return deg / 180. * np.pi


def rad2deg(rad):
    return rad * 180. / np.pi

class Pbh(object):
    # Class for one run
    def __init__(self):
        # the cut on -2lnL, consider smaller values accepted for events coming from the same centroid
        # selected based on sims at 90% efficiency
        #self.ll_cut = -9.5
        self.ll_cut = -8.6
        #self.ll_cut_dict = {2:-9.11,3:-9.00,4:-9.01, 5:-9.06, 6:-9.12, 7:-9.16, 8:-9.19, 9:-9.21, 10:-9.25}
        # Dec=0 after cos correction
        #self.ll_cut_dict = {2:-8.81,3:-8.69,4:-8.80, 5:-8.82, 6:-8.85, 7:-8.90, 8:-8.92, 9:-8.98, 10:-8.99}
        # Dec=80 after cos correction
        #self.ll_cut_dict = {2:-8.81,3:-8.72,4:-8.76, 5:-8.83, 6:-8.86, 7:-8.88, 8:-8.95, 9:-8.96, 10:-8.99}
        # Dec=80 after cos correction, from fit
        #self.ll_cut_dict = {2:-8.83,3:-8.73,4:-8.76, 5:-8.81, 6:-8.86, 7:-8.90, 8:-8.94, 9:-8.97, 10:-8.99}
        # Dec=80 after cos correction, from fit, new cumtrapz integration
        #self.ll_cut_dict = {2:-8.637,3:-8.55,4:-8.564, 5:-8.614, 6:-8.656, 7:-8.7, 8:-8.737, 9:-8.767, 10:-8.794}
        # Dec=80 after cos correction, new cumtrapz integration, theta2 instead of theta
        #self.ll_cut_dict = {2:-6.68,3:-6.74,4:-6.71, 5:-6.76, 6:-6.8, 7:-6.84, 8:-6.88, 9:-6.92, 10:-6.96}
        #new cuts using scrambled data (2017-06-13) 6 runs mean:
        self.ll_cut_dict = {2:-6.95,3:-6.95,4:-6.96, 5:-6.99, 6:-7.03, 7:-7.07, 8:-7.12, 9:-7.16, 10:-7.18}



        # set the hard coded PSF width table from the hyperbolic secant function
        # 4 rows are Energy bins 0.08 to 0.32 TeV (row 0), 0.32 to 0.5 TeV, 0.5 to 1 TeV, and 1 to 50 TeV
        # 3 columns are Elevation bins 50-70 (column 0), 70-80 80-90 degs
        self.psf_lookup = np.zeros((4, 3)).astype('float')
        self.E_grid = np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.EL_grid = np.array([50.0, 70.0, 80., 90.])
        # for later reference
        # self.E_bins=np.digitize(self.BDT_ErecS, self.E_grid)-1
        #self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
        #  0.08 to 0.32 TeV
        #  the 3 elements are Elevation 50-70, 70-80 80-90 degs
        self.psf_lookup[0, :] = np.array([0.052, 0.051, 0.05])
        #  0.32 to 0.5 TeV
        self.psf_lookup[1, :] = np.array([0.047, 0.042, 0.042])
        #   0.5 to 1 TeV
        self.psf_lookup[2, :] = np.array([0.041, 0.035, 0.034])
        #   1 to 50 TeV
        self.psf_lookup[3, :] = np.array([0.031, 0.028, 0.027])
        self._burst_dict = {}  #{"Burst #": [event # in this burst]}, for internal use
        self.VERITAS_deadtime = 0.33e-3  # 0.33ms
        self.runNum = 0
        #self.data_dir = '/raid/reedbuck/qfeng/pbh/data'
        self.data_dir = '/a/data/tehanu/qifeng/pbh/EDroot_files/'
        #self.data_dir = '/global/cscratch1/sd/qifeng/data/pbh_data/'

    def read_photon_list(self, ts, RAs, Decs, Es, ELs):
        N_ = len(ts)
        assert N_ == len(RAs) and N_ == len(Decs) and N_ == len(Es) and N_ == len(ELs), \
            "Make sure input lists (ts, RAs, Decs, Es, ELs) are of the same dimension"
        columns = ['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'psfs', 'burst_sizes', 'fail_cut']
        df_ = pd.DataFrame(np.array([np.zeros(N_)] * len(columns)).T,
                           columns=columns)
        df_.ts = ts
        df_.RAs = RAs
        df_.Decs = Decs
        df_.Es = Es
        df_.ELs = ELs
        # df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        df_.burst_sizes = np.ones(N_)
        #self.photon_df = df_
        #if event.Energy<E_lo_cut or event.Energy>E_hi_cut or event.TelElevation<EL_lo_cut:
        #    df_.fail_cut.at[i] = 1
        #    continue
        df_.fail_cut = np.zeros(N_)
        #clean events that did not pass cut:
        self.photon_df = df_[df_.fail_cut == 0]

        self.get_psf_lists()

    def readEDfile(self, runNum=None, filename=None, dir=None):
        if dir is None:
            dir=self.data_dir
        self.runNum = runNum
        self.filename = str(dir)+"/"+str(runNum) + ".anasum.root"
        if not os.path.isfile(self.filename) and filename is not None:
            if os.path.isfile(filename):
                self.filename = filename
        self.Rfile = ROOT.TFile(self.filename, "read");

    def get_TreeWithAllGamma(self, runNum=None, E_lo_cut=0.08, E_hi_cut=50.0, EL_lo_cut=50.0, distance_upper_cut=1.5, nlines=None):
        """
        :param runNum:
        :return: nothing but fills photon_df, except photon_df.burst_sizes
        """
        if not hasattr(self, 'Rfile'):
            print("No file has been read.")
            if runNum is not None:
                try:
                    self.readEDfile(runNum=runNum)
                    print("Read file " + self.filename + "...")
                except:
                    print("Can't read file with runNum " + str(runNum))
                    raise
            else:
                print("Run self.readEDfile(\"rootfile\") first; or provide a runNum")
                raise
        all_gamma_treeName = "run_" + str(self.runNum) + "/stereo/TreeWithAllGamma"
        # pointingData_treeName = "run_"+str(self.runNum)+"/stereo/pointingDataReduced"
        all_gamma_tree = self.Rfile.Get(all_gamma_treeName)
        #pointingData = self.Rfile.Get(pointingData_treeName)
        #ptTime=[]
        #for ptd in pointingData:
        #    ptTime.append(ptd.Time);
        #ptTime=np.array(ptTime)
        #columns=['runNumber','eventNumber', 'MJD', 'Time', 'Elevation', ]
        columns = ['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'psfs', 'burst_sizes', 'fail_cut']
        # EA is a function of energy, zenith, wobble offset, optical efficiency, and reconstructed camera coords
        if nlines is not None:
            N_ = nlines
        else:
            N_ = all_gamma_tree.GetEntries()
        df_ = pd.DataFrame(np.array([np.zeros(N_)] * len(columns)).T,
                           columns=columns)
        ###QF short breaker:
        #breaker = 0
        #self.alltimes for the use of scramble (will be in random order, sort it to get true all times)
        self.all_times = np.zeros(N_)
        i_gamma = 0
        for i, event in enumerate(all_gamma_tree):
            #if nlines is not None:
            #    if breaker >= nlines:
            #        break
            #    breaker += 1
            #time_index=np.argmax(ptTime>event.Time)
            #making cut:
            #this is quite essential to double check!!!
            self.all_times[i] = event.timeOfDay
            distance = np.sqrt(event.Xderot * event.Xderot + event.Yderot * event.Yderot)
            if event.Energy < E_lo_cut or event.Energy > E_hi_cut or event.TelElevation < EL_lo_cut or event.IsGamma==0 or distance>distance_upper_cut:
                df_.fail_cut.at[i] = 1
                continue
            i_gamma += 1
            # fill the pandas dataframe
            df_.MJDs[i] = event.dayMJD
            #df_.eventNumber[i] = event.eventNumber
            df_.ts[i] = event.timeOfDay
            df_.RAs[i] = event.GammaRA
            df_.Decs[i] = event.GammaDEC
            df_.Es[i] = event.Energy
            df_.ELs[i] = event.TelElevation

        print("There are %d events, %d of which are gamma-like and pass cuts" % (N_,i_gamma))
        self.N_all_events = N_
        self.N_gamma_events = i_gamma
        #df_.coords = np.concatenate([df_.RAs.reshape(N_,1), df_.Decs.reshape(N_,1)], axis=1)
        df_.psfs = np.zeros(N_)
        # by def all events are at least a singlet
        df_.burst_sizes = np.ones(N_)
        #self.photon_df = df_
        #clean events that did not pass cut:
        self.photon_df = df_[df_.fail_cut == 0]
        #reindexing
        self.photon_df.index = range(self.photon_df.shape[0])

        self.get_psf_lists()

        #If
        #df = df[df.line_race.notnull()]

        ###QF
        #print self.photon_df.head()

    def getRunSummary(self):
        tRunSumName = "total_1/stereo/tRunSummary"
        tRunSummary = self.Rfile.Get(tRunSumName)
        for tR in tRunSummary:
            self.tOn = tR.tOn
            self.Rate = tR.Rate
            self.RateOff = tR.RateOff
            self.DeadTimeFracOn = tR.DeadTimeFracOn
            self.TargetRA = tR.TargetRA
            self.TargetDec = tR.TargetDec
            self.TargetRAJ2000 = tR.TargetRAJ2000
            self.TargetDecJ2000 = tR.TargetDecJ2000
            break
        self.total_time_year = (self.tOn*(1.-self.DeadTimeFracOn))/31536000.
        ea_Name = "run_" + str(self.runNum) + "/stereo/EffectiveAreas/gMeanEffectiveArea"
        ea = self.Rfile.Get(ea_Name);
        self.EA = np.zeros((ea.GetN(), 2))
        for i in range(ea.GetN()):
            self.EA[i,0] = ea.GetX()[i]
            self.EA[i,1] = ea.GetY()[i]

        accept_Name = "run_" + str(self.runNum) + "/stereo/RadialAcceptances/fAccZe_0"
        self.accept = self.Rfile.Get(accept_Name);
        #use self.accept.Eval(x) to get y, or just self.accept(x)

    def scramble(self, copy=True, all_events=True):
        if not hasattr(self, 'photon_df'):
            print("Call get_TreeWithAllGamma first...")
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        if all_events:
            #shuffle among the arrival time of all events
            random.shuffle(self.all_times)
            ts_ = self.all_times[:self.N_gamma_events]
        else:
            ts_ = self.photon_df.ts.values
            random.shuffle(ts_)
        self.photon_df.at[:, 'ts'] = ts_
        # re-init _burst_dict for counting
        self._burst_dict = {}
        # print self.photon_df.head()
        #print self.photon_df.ts.shape, self.photon_df.ts
        #sort!
        if pd.__version__>'0.18':
            self.photon_df = self.photon_df.sort_values('ts')
        else:
            self.photon_df=self.photon_df.sort('ts')
        return ts_

    def t_rando(self, copy=True, rate="avg", all_events=True):
        """
        throw Poisson distr. ts based on the original ts,
        use 1/delta_t as the expected Poisson rate for each event
        """
        if not hasattr(self, 'photon_df'):
            print("Call get_TreeWithAllGamma first...")
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        if rate=="cell":
            delta_ts = np.diff(self.photon_df.ts)
        #for i, _delta_t in enumerate(delta_ts):
        if all_events:
            N = self.N_all_events
            self.rando_all_times = np.zeros(N)
            rate_expected= N*1.0/(self.all_times[-1]-self.all_times[0])
        else:
            N = self.photon_df.shape[0]
            rate_expected= N*1.0/(self.photon_df.ts.values[-1]-self.photon_df.ts.values[0])
        print("Mean expected rate is %.2f" % rate_expected)
        for i in range(N-1):
            if rate=="cell":
                rate_expected = 1. / delta_ts[i]
            #elif rate=="avg":
            if all_events:
                _rando_delta_t = np.random.exponential(1./rate_expected)
                inf_loop_preventer = 0
                inf_loop_bound = 100
                while _rando_delta_t < self.VERITAS_deadtime:
                    _rando_delta_t = np.random.exponential(1./rate_expected)
                    inf_loop_preventer += 1
                    if inf_loop_preventer > inf_loop_bound:
                        print("Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,")
                        print("you'd better check your time unit or something...")
                self.rando_all_times[i + 1] = self.rando_all_times[i] + _rando_delta_t
            else:
                # draw a rando!
                _rando_delta_t = np.random.exponential(1./rate_expected)
                inf_loop_preventer = 0
                inf_loop_bound = 100
                while _rando_delta_t < self.VERITAS_deadtime:
                    _rando_delta_t = np.random.exponential(1./rate_expected)
                    inf_loop_preventer += 1
                    if inf_loop_preventer > inf_loop_bound:
                        print("Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,")
                        print("you'd better check your time unit or something...")
                self.photon_df.at[i + 1, 'ts'] = self.photon_df.ts[i] + _rando_delta_t
        if all_events:
            random.shuffle(self.rando_all_times)
            for i, in range(self.photon_df.shape[0]):
                self.photon_df.at[i, 'ts'] = self.rando_all_times[i]
        #naturally sorted
        # re-init _burst_dict for counting
        self._burst_dict = {}
        return self.photon_df.ts


    #@autojit
    def psf_func(self, theta2, psf_width, N=100):
        return 1.7142 * N / 2. / np.pi / (psf_width ** 2) / np.cosh(np.sqrt(theta2) / psf_width)
        # equivalently:
        #return (stats.hypsecant.pdf(np.sqrt(theta2s)/psf_width)*1.7142/2./psf_width**2)

    def psf_cdf(self, psf_width, fov=1.75):
        """
        :param psf_width: same as psf_func
        :param fov: given so that we calculate cdf from 0 to fov
        :return:
        """
        _thetas = np.arange(0, fov, 0.001)
        _theta2s = _thetas ** 2
        #cdf = np.cumsum(self.psf_func(theta2s, psf_width, N=1))
        cdf = integrate.cumtrapz(self.psf_func(_theta2s, psf_width, N=1), _thetas, initial=0)
        cdf = cdf / np.max(cdf)
        return cdf

    # use hard coded width table from the hyperbolic secant function
    def get_psf(self, E=0.1, EL=80):
        E_bin = np.digitize(E, self.E_grid, right=True) - 1
        EL_bin = np.digitize(EL, self.EL_grid, right=True) - 1
        return self.psf_lookup[E_bin, EL_bin]

    #@autojit
    def get_psf_lists(self):
        """
        This thing is slow...
        :return: nothing but filles photon_df.psfs, a number that is repeatedly used later
        """
        if not hasattr(self, 'photon_df'):
            print("Call get_TreeWithAllGamma first...")
        ###QF:
        print("getting psf")
        for i, EL_ in enumerate(self.photon_df.ELs.values):
            #self.photon_df.psfs.at[i] = self.get_psf(E=self.photon_df.loc[i, 'Es'], EL=EL_)
            self.photon_df.at[i, 'psfs'] = self.get_psf(E=self.photon_df.at[i, 'Es'], EL=EL_)
            #if i%10000==0:
            #    print i, "events got psfs"
            #    print self.photon_df.at[i, 'Es'], EL_
            #    print self.photon_df.psfs[i]
            #if self.photon_df.psfs[i] is None:
            #    print "Got a None psf, energy is ", self.photon_df.at[i, 'Es'], "EL is ", EL_
            #print "PSF,", self.photon_df.psfs.at[i]

    #@autojit
    def get_angular_distance(self, coord1, coord2):
        """
        coord1 and coord2 are in [ra, dec] format in degrees
        """
        return rad2deg(np.arccos(np.sin(deg2rad(coord1[1])) * np.sin(deg2rad(coord2[1]))
                                 + np.cos(deg2rad(coord1[1])) * np.cos(deg2rad(coord2[1])) *
                                 np.cos(deg2rad(coord1[0]) - deg2rad(coord2[0]))))

    #@autojit
    def get_all_angular_distance(self, coords, cent_coord):
        assert coords.shape[1] == 2
        dists = np.zeros(coords.shape[0])
        for i, coord in enumerate(coords):
            dists[i] = self.get_angular_distance(coord, cent_coord)
        return dists

    def gen_one_random_coords_projected_plane(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        # the dec of this small circle should be in the range of [cent_dec - theta, cent_dec + theta]
        delta_dec = (np.random.random() * 2. -1.) * theta
        _dec = cent_coord[1] + delta_dec
        # Note that dec is 90 deg - theta in spherical coordinates
        _ra = cent_coord[0] + rad2deg( np.arccos ( np.cos(deg2rad(theta)) * (1./np.cos(deg2rad(90.-cent_coord[1]))) * (1./np.cos(deg2rad(90.-_dec))) \
                                                   - np.tan(deg2rad(90.-cent_coord[1])) *  np.tan(deg2rad(90.-_dec)) ) )
        #_ra = cent_coord[0] + rad2deg( np.arccos ( np.cos(deg2rad(theta)) * (1./np.cos(deg2rad(cent_coord[1]))) * (1./np.cos(deg2rad(_dec))) \
        #                                           - np.tan(deg2rad(cent_coord[1])) *  np.tan(deg2rad(_dec)) ) )
        return np.array([_ra, _dec])

    def gen_one_random_coords(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        _phi = np.random.random() * np.pi * 2.
        #_ra = cent_coord[0] + np.sin(_phi) * theta
        _ra = cent_coord[0] + np.sin(_phi) * theta / np.cos(deg2rad(cent_coord[1]))
        _dec = cent_coord[1] + np.cos(_phi) * theta
        return np.array([_ra, _dec])

    def gen_one_random_theta(self, psf_width, prob="psf", fov=1.75):
        """
        :prob can be either "psf" that uses the hyper-sec function, or "uniform", or "gauss"
        """
        if prob.lower() == "psf" or prob == "hypersec" or prob == "hyper secant":
            #_rand_theta = np.random.random()*fov
            _rand_test_cdf = np.random.random()
            #_thetas = np.arange(0, fov, 0.001)
            #_theta2s = _thetas ** 2
            _theta2s = np.arange(0, fov ** 2, 1e-4)
            _thetas = np.sqrt(_theta2s)
            _psf_pdf = self.psf_func(_theta2s, psf_width, N=1)
            #_cdf = np.cumsum(_psf_pdf - np.min(_psf_pdf))
            #_cdf = integrate.cumtrapz(_psf_pdf, _thetas, initial=0)
            _cdf = integrate.cumtrapz(_psf_pdf, _theta2s, initial=0)
            _cdf = _cdf / np.max(_cdf)
            #y_interp = np.interp(x_interp, x, y)
            _theta2 = np.interp(_rand_test_cdf, _cdf, _theta2s)
            return np.sqrt(_theta2)
        elif prob.lower() == "uniform" or prob == "uni":
            return np.random.random() * fov
        #gauss may have a caveat as this is not important
        elif prob.lower() == "gauss" or prob == "norm" or prob.lower() == "gaussian":
            return abs(np.random.normal()) * fov
        else:
            return "Input prob value not supported"

    def gen_one_random_theta_simon_method(self, psf_width, prob="psf", fov=1.75):

        def fControl(x, psf_width=0.05):
            return 4. * psf_width / np.pi * np.arctanh(np.tan(np.pi * x / 4.))
            # return 4.*psf_width/np.pi*np.arctanh(np.exp(np.pi*x/4.))

        def fC_Function(x, psf_width=0.05):
            return 1. / (psf_width * np.cosh(np.sqrt(x) * 0.5 / psf_width))

        def psf_pdf_simon(x, psf_width=0.05):
            return 1.7149 / (2 * np.pi * psf_width * np.cosh(np.sqrt(x) * 1.0 / psf_width))

        z0 = 9999.
        pdf_y0 = 0.
        y0 = 0.
        while z0 > pdf_y0:
            _rand_ = np.random.random()
            y0 = fControl(_rand_, psf_width=psf_width)
            c_y0 = fC_Function(y0, psf_width=psf_width)
            z0 = c_y0 * np.random.random()
            # z0 = c_y0 * _rand_
            pdf_y0 = psf_pdf_simon(y0, psf_width)
            # pdf_y0 = psf_func(y0, psf_width, N=1)
        return y0

    #@autojit
    def centroid_log_likelihood(self, cent_coord, coords, psfs):
        """
        returns ll=-2*ln(L)
        """
        ll = 0
        dists = self.get_all_angular_distance(coords, cent_coord)
        theta2s = dists ** 2
        #self.psf_func(theta2, psf_width, N=1)
        #ll = -2.* np.log(  1.7142 * N / 2. / np.pi / (psf_width ** 2) / np.cosh(np.sqrt(theta2) / psf_width)  )
        ll = -2. * np.sum(np.log(psfs)) + np.sum(np.log(1. / np.cosh(np.sqrt(theta2s) / psfs)))
        ll += psfs.shape[0] * np.log(1.7142 / np.pi)
        ll = -2. * ll
        #return ll
        #Normalized by the number of events!
        return ll / psfs.shape[0]

    #@autojit
    def minimize_centroid_ll(self, coords, psfs):
        init_centroid = np.mean(coords, axis=0)
        results = minimize(self.centroid_log_likelihood, init_centroid, args=(coords, psfs), method='L-BFGS-B')
        centroid = results.x
        ll_centroid = self.centroid_log_likelihood(centroid, coords, psfs)
        return centroid, ll_centroid

    def search_angular_window(self, coords, psfs, slice_index):
        # Determine if N_evt = coords.shape[0] events are accepted to come from one direction
        # slice_index is the numpy array slice of the input event numbers, used for _burst_dict
        # return: A) centroid, likelihood, and a list of event numbers associated with this burst,
        #            given that a burst is found, or the input has only one event
        #         B) centroid, likelihood, a list of event numbers excluding the outlier, the outlier event number
        #            given that we reject the hypothesis of a burst
        ###QF
        #print coords, slice_index
        assert coords.shape[0] == slice_index.shape[0], "coords shape " + coords.shape[0] + " and slice_index shape " + \
                                                        slice_index.shape[0] + " are different"
        if slice_index.shape[0] == 0:
            #empty
            return None, None, None, None
        if slice_index.shape[0] == 1:
            #one event:
            return coords, 1, np.array([1])
        centroid, ll_centroid = self.minimize_centroid_ll(coords, psfs)
        if slice_index.shape[0] in self.ll_cut_dict.keys():
            psf_ll_cut=self.ll_cut_dict[slice_index.shape[0]]
        else:
            psf_ll_cut=self.ll_cut
        #if ll_centroid <= self.ll_cut:
        if ll_centroid <= psf_ll_cut:
            # likelihood passes cut
            # all events with slice_index form a burst candidate
            return centroid, ll_centroid, slice_index
        else:
            # if not accepted, find the worst offender and
            # return the better N_evt-1 events and the outlier event
            dists = self.get_all_angular_distance(coords, centroid)
            outlier_index = np.argmax(dists)
            mask = np.ones(len(dists), dtype=bool)
            mask[outlier_index] = False
            #better_coords, better_psfs, outlier_coords, outlier_psfs = coords[mask,:], psfs[mask],\
            #                                                           coords[outlier_index,:], psfs[outlier_index]

            #better_centroid, better_ll_centroid, better_burst_sizes = self.search_angular_window(better_coords, better_psfs)

            #return centroid, ll_centroid, coords[mask,:], psfs[mask], coords[outlier_index,:], psfs[outlier_index]
            #return centroid, ll_centroid, slice_index[mask], slice_index[outlier_index]
            ###QF
            #print "mask", mask
            #print "outlier", outlier_index
            #print "slice_index", slice_index, type(slice_index)
            #print "search_angular_window returning better events", slice_index[mask]
            #print "returning outlier events", slice_index[outlier_index]
            return centroid, ll_centroid, slice_index[mask], slice_index[outlier_index]

    def search_time_window(self, window_size=1, verbose=False):
        """
        Start a burst search for the given window_size in photon_df
        _burst_dict needs to be clean for a new scramble
        :param window_size: in the unit of second
        :return: burst_hist, in the process 1) fill self._burst_dict, and
                                            2) fill self.photon_df.burst_sizes through burst counting; and
                                            3) fill self.photon_df.burst_sizes
        """
        assert hasattr(self,
                       'photon_df'), "photon_df doesn't exist, read data first (read_photon_list or get_TreeWithAllGamma)"
        if len(self._burst_dict) != 0:
            print("You started a burst search while there are already things in _burst_dict, now make it empty")
            self._burst_dict = {}
        previous_window_start = -1.0
        previous_window_end = -1.0
        previous_singlets = np.array([])
        previous_non_singlets = np.array([])
        # Master event loop:
        for t in self.photon_df.ts:
            if verbose:
                print("Starting at the event at %.5f" % t)
            if previous_window_start == -1.0:
                #first event:
                previous_window_start = t
                previous_window_end = t + window_size
            else:
                #below just see if there are new events in the extra time interval after the previous_window_end
                new_event_slice_tuple = np.where((self.photon_df.ts >= previous_window_end) & (self.photon_df.ts < (t + window_size)))
                previous_window_start = t
                previous_window_end = t + window_size
                if len(new_event_slice_tuple[0]) == 0:
                    #no new events in the extra window, continue
                    if verbose:
                        print("no new events found, skipping to next event")
                    continue

            #1. slice events between t and t+window_size
            slice_index = np.where((self.photon_df.ts >= t) & (self.photon_df.ts < (t + window_size)))

            #2. remove singlets
            slice_index, singlet_slice_index = self.singlet_remover(np.array(slice_index[0]), verbose=verbose)

            if slice_index is None:
                if verbose:
                    print("All events are singlet in this time window")
                #All events are singlet, removed all
                continue
            elif len(slice_index) == 0:
                if verbose:
                    print("All events are singlet in this time window")
                #All events are singlet, removed all
                continue

            #check if all new events are singlets,
            # if so the current better_events list should be contained in the previous one.
            #if np.in1d(singlet_slice_index, previous_singlets).all():
            if np.in1d(slice_index, previous_non_singlets).all():
                previous_singlets=singlet_slice_index
                previous_non_singlets = slice_index
                continue
            previous_singlets=singlet_slice_index
            previous_non_singlets=slice_index

            slice_index = tuple(slice_index[:, np.newaxis].T)

            _N = self.photon_df.ts.values[slice_index].shape[0]
            #sanity check
            if _N < 1:
                #All events are singlet, removed all
                if verbose:
                    print("Can't reach here")
                continue
            elif _N == 1:
                #a sparse window
                #self.photon_df.burst_sizes[slice_index] = 1
                #print "L367", slice_index
                #self.photon_df.at[slice_index[0], 'burst_sizes'] = 1
                if verbose:
                    print("Can't reach here")
                continue

            #3. burst searching (in angular window)
            burst_events, outlier_events = self.search_event_slice(np.array(slice_index[0]))
            if outlier_events is None:
                #All events of slice_index form a burst, no outliers; or all events are singlet
                if verbose:
                    print("All events form 1 burst")
                continue
            #elif len(outlier_events)==1:
            elif outlier_events.shape[0] == 1:
                #A singlet outlier
                #self.photon_df.burst_sizes[outlier_events[0]] = 1
                #print "L378", outlier_events, outlier_events[0]
                #self.photon_df.at[outlier_events[0], 'burst_sizes'] = 1
                continue
            else:
                #If there is a burst of a subset of events, it's been taken care of, now take care of the outlier slice (which has >1 events reaching here)
                outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(outlier_events)

                while outlier_of_outlier_events is not None:
                    ###QF
                    if verbose:
                        print("looping through the outliers ")
                    #loop until no outliers are left unprocessed
                    if len(outlier_of_outlier_events) <= 1:
                        #self.photon_df.burst_sizes[outlier_of_outlier_events[0]] = 1
                        outlier_of_outlier_events = None
                        break
                    else:
                        # more than 1 outliers to process,
                        # update outlier_of_outlier_events and repeat the while loop
                        outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(
                            outlier_of_outlier_events)
        # the end of master event loop, self._burst_dict is filled
        # now count bursts and fill self.photon_df.burst_sizes:
        if verbose:
            print("Counting bursts")
        #_burst_dict = self._burst_dict.copy()
        self.duplicate_burst_dict()
        # initialize burst sizes
        self.photon_df.at[:, 'burst_sizes'] = 1

        #Note now self._burst_dict will be cleared!!
        self.burst_counting()
        burst_hist = self.get_burst_hist()
        if verbose:
            print("Found bursts: %s"%burst_hist)
        #return self.photon_df.burst_sizes
        return burst_hist, self.burst_dict

    #@autojit
    def singlet_remover(self, slice_index, verbose=False):
        """
        :param slice_index: a np array of events' indices in photon_df
        :return: new slice_index with singlets (no neighbors in a radius of 5*psf) removed, and a slice of singlets;
                 return None and input slice if all events are singlets
        """
        if slice_index.shape[0] == 1:
            #one event, singlet by definition:
            #return Nones
            #self.photon_df.at[slice_index[0], 'burst_sizes'] = 1
            return None, slice_index
        N_ = self.photon_df.shape[0]
        slice_tuple = tuple(slice_index[:, np.newaxis].T)
        coord_slice = np.concatenate([self.photon_df.RAs.values.reshape(N_, 1), self.photon_df.Decs.values.reshape(N_, 1)], axis=1)[
            slice_tuple]
        psf_slice = self.photon_df.psfs.values[slice_tuple]
        #default all events are singlet
        mask_ = np.zeros(slice_index.shape[0], dtype=bool)
        #use a dict of {event_num:neighbor_found} to avoid redundancy
        none_singlet_dict = {}
        for i in range(slice_index.shape[0]):
            if slice_index[i] in none_singlet_dict:
                #already knew not a singlet
                continue
            else:
                psf_5 = psf_slice[i] * 5.0
                for j in range(slice_index.shape[0]):
                    if j == i:
                        #self
                        continue
                    elif self.get_angular_distance(coord_slice[i], coord_slice[j]) < psf_5:
                        #decide this pair isn't singlet
                        none_singlet_dict[slice_index[i]] = slice_index[j]
                        none_singlet_dict[slice_index[j]] = slice_index[j]
                        mask_[i] = True
                        mask_[j] = True
                        continue
        ###QF
        if verbose:
            print("removed %d singlet" % sum(mask_==False))
            print("%d good evts" % slice_index[mask_].shape[0])
        return slice_index[mask_], slice_index[np.invert(mask_)]

    def search_event_slice(self, slice_index):
        """
        _burst_dict needs to be clean before starting a new scramble
        :param slice_index: np array of indices of the events in photon_df that the burst search is carried out upon
        :return: np array of indices of events that are in a burst, indices of outliers (None if no outliers);
                 in the process fill self._burst_dict for later burst counting
        """
        N_ = self.photon_df.shape[0]
        ###QF
        #print "Slice"
        #print slice_index
        #print "Type"
        #print type(slice_index)
        #print "tuple Slice"
        #print tuple(slice_index)
        #print "length", len(tuple(np.array(slice_index)[:,np.newaxis].T))
        #print "Coords"
        #print "Shape"
        #print np.concatenate([self.photon_df.RAs.reshape(N_,1), self.photon_df.Decs.reshape(N_,1)], axis=1).shape
        #print np.concatenate([self.photon_df.RAs.values.reshape(N_,1), self.photon_df.Decs.values.reshape(N_,1)], axis=1)[tuple(slice_index[:,np.newaxis].T)]
        #print "PSFs"
        #print self.photon_df.psfs.values[tuple(slice_index[:,np.newaxis].T)]

        #First remove singlet
        #slice_index = self.singlet_remover(slice_index)
        #print slice_index
        if slice_index.shape[0] == 0:
            #all singlets, no bursts, and don't need to check for outliers, go to next event
            return None, None

        ang_search_res = self.search_angular_window(
            np.concatenate([self.photon_df.RAs.values.reshape(N_, 1), self.photon_df.Decs.values.reshape(N_, 1)], axis=1)[
                tuple(slice_index[:, np.newaxis].T)], self.photon_df.psfs.values[tuple(slice_index[:, np.newaxis].T)],
            slice_index)
        outlier_evts = []

        if len(ang_search_res) == 3:
            # All events with slice_index form 1 burst
            centroid, ll_centroid, burst_events = ang_search_res
            self._burst_dict[len(self._burst_dict) + 1] = burst_events
            #count later
            #self.photon_df.burst_sizes[slice_index] = len(burst_events)
            #burst_events should be the same as slice_index
            return burst_events, None
        else:
            while (len(ang_search_res) == 4):
                # returned 4 meaning no bursts, and the input has more than one events, shall continue
                # this loop breaks when a burst is found or only one event is left, in which case return values has a length of 3
                better_centroid, better_ll_centroid, _better_events, _outlier_events = ang_search_res
                outlier_evts.append(_outlier_events)
                ###QF
                #print tuple(_better_events), _better_events
                #better_coords = np.concatenate([self.photon_df.RAs.reshape(N_,1), self.photon_df.Decs.reshape(N_,1)], axis=1)[tuple(_better_events)]
                better_coords = \
                    np.concatenate([self.photon_df.RAs.values.reshape(N_, 1), self.photon_df.Decs.values.reshape(N_, 1)], axis=1)[
                        (_better_events)]
                #print "in search_event_slice, candidate coords and psfs: ", better_coords, self.photon_df.psfs.values[(_better_events)]
                ang_search_res = self.search_angular_window(better_coords,
                                                            self.photon_df.psfs.values[(_better_events)],
                                                            _better_events)
            # Now that while loop broke, we have a good list and a bad list
            centroid, ll_centroid, burst_events = ang_search_res
            if burst_events.shape[0] == 1:
                # No burst in slice_index found
                #count later
                #self.photon_df.burst_sizes[burst_events[0]] = 1
                return burst_events, np.array(outlier_evts)
            else:
                # A burst with a subset of events of slice_index is found
                self._burst_dict[len(self._burst_dict) + 1] = burst_events
                #self.photon_df.burst_sizes[tuple(burst_events)] = len(burst_events)
                return burst_events, np.array(outlier_evts)


    def duplicate_burst_dict(self):
        #if you want to keep the original burst_dict
        self.burst_dict = self._burst_dict.copy()
        return self.burst_dict

    #@autojit
    def burst_counting_recur(self):
        """
        :return: nothing but fills self.photon_df.burst_sizes, during the process self._burst_dict is emptied!
        """
        # Only to be called after self._burst_dict is filled
        # Find the largest burst
        largest_burst_number = max(self._burst_dict, key=lambda x: len(set(self._burst_dict[x])))
        for evt in self._burst_dict[largest_burst_number]:
            # Assign burst size to all events in the largest burst
            self.photon_df.at[evt, 'burst_sizes'] = self._burst_dict[largest_burst_number].shape[0]
            #self.photon_df.burst_sizes[evt] = len(self._burst_dict[largest_burst_number])
            for key in self._burst_dict.keys():
                # Now delete the assigned events in all other candiate bursts to avoid double counting
                if evt in self._burst_dict[key] and key != largest_burst_number:
                    #self._burst_dict[key].remove(evt)
                    self._burst_dict[key] = np.delete(self._burst_dict[key], np.where(self._burst_dict[key] == evt))
        # Delete the largest burst, which is processed above
        self._burst_dict.pop(largest_burst_number, None)
        # repeat while there are unprocessed bursts in _burst_dict
        if len(self._burst_dict) >= 1:
            self.burst_counting_recur()

    def burst_counting(self):
        """
        :return: nothing but fills self.photon_df.burst_sizes, during the process self._burst_dict is emptied!
        """
        # Only to be called after self._burst_dict is filled
        while len(self._burst_dict) >= 1:
            # Find the largest burst
            largest_burst_number = max(self._burst_dict, key=lambda x: len(set(self._burst_dict[x])))
            for evt in self._burst_dict[largest_burst_number]:
                # Assign burst size to all events in the largest burst
                self.photon_df.at[evt, 'burst_sizes'] = self._burst_dict[largest_burst_number].shape[0]
                #self.photon_df.burst_sizes[evt] = len(self._burst_dict[largest_burst_number])
                for key in self._burst_dict.keys():
                    # Now delete the assigned events in all other candiate bursts to avoid double counting
                    if evt in self._burst_dict[key] and key != largest_burst_number:
                        #self._burst_dict[key].remove(evt)
                        self._burst_dict[key] = np.delete(self._burst_dict[key], np.where(self._burst_dict[key] == evt))
            # Delete the largest burst, which is processed above
            self._burst_dict.pop(largest_burst_number, None)
            # repeat while there are unprocessed bursts in _burst_dict
            #if len(self._burst_dict) >= 1:
            #    self.burst_counting()

    def burst_counting_fractional(self):
        """
        :return: nothing but fills self.photon_df.burst_sizes, during the process self._burst_dict is emptied!
        """
        # Only to be called after self._burst_dict is filled
        while len(self._burst_dict) >= 1:
            # Find the largest burst
            largest_burst_number = max(self._burst_dict, key=lambda x: len(set(self._burst_dict[x])))
            for evt in self._burst_dict[largest_burst_number]:
                # Assign burst size to all events in the largest burst
                self.photon_df.at[evt, 'burst_sizes'] = self._burst_dict[largest_burst_number].shape[0]
                #self.photon_df.burst_sizes[evt] = len(self._burst_dict[largest_burst_number])
                for key in self._burst_dict.keys():
                    # Now delete the assigned events in all other candiate bursts to avoid double counting
                    if evt in self._burst_dict[key] and key != largest_burst_number:
                        #self._burst_dict[key].remove(evt)
                        self._burst_dict[key] = np.delete(self._burst_dict[key], np.where(self._burst_dict[key] == evt))
            # Delete the largest burst, which is processed above
            self._burst_dict.pop(largest_burst_number, None)

    def get_burst_hist(self):
        burst_hist = {}
        for i in np.unique(self.photon_df.burst_sizes.values):
            burst_hist[i] = np.sum(self.photon_df.burst_sizes.values == i) / i
        return burst_hist

    def sig_burst_search(self, window_size=1, verbose=False):
        _sig_burst_hist, _sig_burst_dict = self.search_time_window(window_size=window_size, verbose=verbose)
        self.sig_burst_hist = _sig_burst_hist.copy()
        self.sig_burst_dict = _sig_burst_dict.copy()
        return self.sig_burst_hist, self.sig_burst_dict

    def estimate_bkg_burst(self, window_size=1, method="scramble", copy=True, n_scramble=10, rando_method="avg",
                           return_burst_dict=False, verbose=False, all=True):
        """
        :param method: either "scramble" or "rando"
        :return:
        """
        #Note that from now on we are CHANGING the photon_df!

        bkg_burst_hists = []
        bkg_burst_dicts = []
        for i in range(n_scramble):
            #bkg_burst_hist = pbh.estimate_bkg_burst(window_size=window_size)
            if method == "scramble":
                self.scramble(copy=copy, all_events=all)
            elif method == "rando":
                self.t_rando(copy=copy, rate=rando_method)
            bkg_burst_hist, bkg_burst_dict = self.search_time_window(window_size=window_size, verbose=verbose)
            bkg_burst_hists.append(bkg_burst_hist.copy())
            if return_burst_dict:
                bkg_burst_dicts.append(bkg_burst_dict.copy())

        self.bkg_burst_hists = bkg_burst_hists

        all_bkg_burst_sizes = set(k for dic in bkg_burst_hists for k in dic.keys())
        #also a dict
        avg_bkg_hist = {}
        #avg_bkg_hist_count = {}
        for key_ in all_bkg_burst_sizes:
            key_ = int(key_)
            for d_ in bkg_burst_hists:
                if key_ in d_:
                    if key_ in avg_bkg_hist:
                        avg_bkg_hist[key_] += d_[key_]
                        #avg_bkg_hist_count[key_] += 1
                    else:
                        avg_bkg_hist[int(key_)] = d_[key_]
                        #avg_bkg_hist_count[int(key_)] = 1

        for k in avg_bkg_hist.keys():
            #avg_bkg_hist[k] /= avg_bkg_hist_count[k]*1.0
            avg_bkg_hist[k] /= n_scramble*1.0

        self.avg_bkg_hist = avg_bkg_hist.copy()
        #return bkg_burst_hists, avg_bkg_hist
        if return_burst_dict:
            return self.avg_bkg_hist, bkg_burst_dicts
        return self.avg_bkg_hist

    def get_residual_hist(self):
        residual_dict={}
        sig_bkg_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        #fill with zero if no burst size count
        for key_ in sig_bkg_burst_sizes:
            key_ = int(key_)
            if key_ not in self.sig_burst_hist:
                self.sig_burst_hist[key_] = 0
            if key_ not in self.avg_bkg_hist:
                self.avg_bkg_hist[key_] = 0
            residual_dict[key_] = self.sig_burst_hist[key_] - self.avg_bkg_hist[key_]
        self.residual_dict = residual_dict
        return residual_dict

    def kT_BH(self, t_window):
        #eq 8.2
        #temperature (energy in the unit of GeV) of the BH
        return 7.8e3*(t_window)**(-1./3)

    def diff_raw_number(self, E, kT_BH):
        #eq 8.1
        #The expected dN/dE of gammas at energy E (unit GeV), temp kT_BH(tau)
        if E<kT_BH:
            n_exp_gamma = 9.e35 * kT_BH**(-1.5) * E**(-1.5)
        else:
            n_exp_gamma = 9.e35 * E**(-3)
        return n_exp_gamma

    def get_integral_expected(self, kT_BH, verbose=False):
        #integrate over EA between energies:
        self.elo = 80.
        self.ehi = 50000.
        #the integral part of eq 8.3 with no acceptance raised to 3/2 power (I^3/2 in eq 8.7); EA normalized to the unit of pc^2
        #The expected # of gammas
        if not hasattr(self,'EA'):
            print("self.EA doesn't exist, reading it now")
            self.getRunSummary()
        #2D array, energy and (dN/dE * EA)
        number_expected = np.zeros((self.EA.shape[0], 2))
        count = 0
        for e_, ea_ in self.EA:
            #print e_, ea_
            diff_n_exp = self.diff_raw_number(10**e_*1000., kT_BH)
            number_expected[count, 0] = 10**e_*1000.
            number_expected[count, 1] = diff_n_exp * ea_ / (3.086e+16**2)
            count += 1
            # 1 pc = 3.086e+16 m
        energy_cut_indices = np.where((number_expected[:, 0]>=self.elo) & (number_expected[:, 0]<=self.ehi))
        integral_expected = np.trapz(number_expected[energy_cut_indices, 1], x=number_expected[energy_cut_indices, 0])
        #This is the "I**(3./2)" in eq 8.7 in Simon's thesis
        integral_expected = integral_expected**(3./2.)
        if verbose:
            print("The value of I in eq 8.7 is %.2f" % integral_expected)
        return integral_expected

    def get_accept_integral(self, integral_limit = 1.5, verbose=False):
        # \int (g(alpha, beta))**(3./2) d(cos(theta)) in eq 8.7
        rad_=np.arange(0,integral_limit,0.001)
        acc_=[]
        for d_ in rad_:
            acc_.append(self.accept(d_))
        accept_int = np.trapz(np.array(acc_)**(3./2) * np.sin(rad_*np.pi/180.), x = rad_)
        if verbose:
            print("The value of the acceptance integral in eq 8.7 is %.2f" % accept_int)
        return accept_int

    def V_eff(self, burst_size, t_window, verbose=False):
        #eq 8.7; time is in the unit of year
        I_Value = self.get_integral_expected(self.kT_BH(t_window))
        rad_Int = self.get_accept_integral()
        effVolume = (1./(8*np.sqrt(np.pi))) * gamma(burst_size - 1.5) / factorial(burst_size) * I_Value * rad_Int #* self.total_time_year
        if verbose:
            print("The value of the effective volume (eq 8.7) is %.2f" % effVolume)
        return effVolume

    def n_excess(self, rho_dot, Veff, verbose=False):
        #eq 8.8, or maybe more appropriately call it n_expected
        if not hasattr(self, 'total_time_year'):
            self.total_time_year = (self.tOn*(1.-self.DeadTimeFracOn))/31536000.
        #n_ex = 1.0 * rho_dot * self.total_time_year * Veff
        # because the burst likelihood cut -9.5 is at 90% CL
        n_ex = 0.9 * rho_dot * self.total_time_year * Veff
        if verbose:
            print("The value of the expected number of bursts (eq 8.9) is %.2f" % n_ex)
        return n_ex

    #@autojit
    def ll(self, n_on, n_off, n_expected):
        #eq 8.13 without the sum
        return -2. * (-1. * n_expected + n_on * np.log(n_off + n_expected))
    """
    def ll(self, n_on, n_off, n_expected, verbose=False):
        #eq 8.13 without the sum
        ll_ = -2. * (-1. * n_expected + n_on * np.log(n_off + n_expected))
        if verbose:
            print("The likelihood value term for the given burst size (eq 8.13 before sum) is %.2f" % ll_)
        return ll_
    """

    def get_significance(self, verbose=False):
        residual_dict = self.residual_dict
        significance = 0
        for b_, excess_ in residual_dict.iteritems():
            err_excess_ = np.sqrt( self.sig_burst_hist[b_] + pow(np.sqrt(10*self.bkg_burst_hists[b_])/10,2) )
            if verbose:
                print("Significance for bin %d has significance %.2f" % (b_, excess_/err_excess_) )
            significance += excess_/err_excess_
        if verbose:
            print("Overall Significance is %d" % significance)
        return significance

    def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=None):
    #def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=100):
        #eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
        if upper_burst_size is None:
            all_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        else:
            all_burst_sizes = range(burst_size_threshold,upper_burst_size+1)
        ll_ = 0.0
        sum_nb = 0
        self.good_burst_sizes=[] # use this to keep burst sizes that are large enough so that $n_b > \sum_b n_{b+1}$
        #for burst_size in all_burst_sizes:
        for burst_size in np.sort(np.array(list(all_burst_sizes)))[::-1]:
        # starting from the largest burst to test whether $n_b > \sum_b n_{b+1}$
            if burst_size >= burst_size_threshold:
                Veff_ = self.V_eff(burst_size, t_window, verbose=verbose)
                n_expected_ = self.n_excess(rho_dot, Veff_, verbose=verbose)
                if burst_size not in self.sig_burst_hist:
                    self.sig_burst_hist[burst_size] = 0
                if burst_size not in self.avg_bkg_hist:
                    self.avg_bkg_hist[burst_size] = 0
                n_on_ = self.sig_burst_hist[burst_size]
                n_off_ = self.avg_bkg_hist[burst_size]
                if n_on_ < sum_nb:
                    print("reached where n_b < \sum_b n_(b+1), at b={}".format(burst_size))
                    print("Stopping")
                    break
                else:
                    self.good_burst_sizes.append(burst_size)
                ll_ += self.ll(n_on_, n_off_, n_expected_)
                sum_nb += n_on_
                if verbose:
                    #print("###############################################################################")
                    print("Adding -2lnL at burst size %d, for search window %.1f and rate density %.1f, so far -2lnL = %.2f" % (burst_size, t_window, rho_dot, ll_))
                    #print("###############################################################################")
        if verbose:
            print("###############################################################################")
            print("-2lnL above burst size %d, for search window %.1f and rate density %.1f is %.2f" % (burst_size_threshold, t_window, rho_dot, ll_))
            print("###############################################################################")
        return ll_

    def get_ll_vs_rho_dot(self, burst_size_thresh, t_window, rho_dots=np.arange(0., 3.e5, 100), verbose=False, upper_burst_size=None):
    #def get_ll_vs_rho_dot(self, burst_size_thresh, t_window, rho_dots=np.arange(0., 3.e5, 100), verbose=False, upper_burst_size=100):
        #plot a vertical slice of Fig 8-4, for a given burst size and search window, scan through rho_dot and plot -2lnL
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        lls_ = np.zeros(rho_dots.shape[0])
        for i, rho_dot_ in enumerate(rho_dots):
            lls_[i] = self.get_ll(rho_dot_, burst_size_thresh, t_window, verbose=verbose, upper_burst_size=upper_burst_size)
        return rho_dots, lls_

    #@autojit
    def get_minimum_ll(self, burst_size, t_window, rho_dots=np.arange(0., 3.e5, 100), return_arrays=True,
                       #verbose=False, upper_burst_size=100):
                       verbose=False, upper_burst_size=None):
        #search rho_dots for the minimum -2lnL
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        min_ll_ = 1.e5
        rho_dot_min_ll_ = -1.0
        if return_arrays:
            lls_ = np.zeros(rho_dots.shape[0])
        i=0
        for rho_dot_ in rho_dots:
            ll_ = self.get_ll(rho_dot_, burst_size, t_window, verbose=verbose, upper_burst_size=upper_burst_size)
            if ll_ < min_ll_:
                min_ll_ = ll_
                rho_dot_min_ll_ = rho_dot_
            if return_arrays:
                lls_[i] = ll_
                i += 1
        if return_arrays:
            return rho_dot_min_ll_, min_ll_, rho_dots, lls_
        return rho_dot_min_ll_, min_ll_

    def get_ul_rho_dot(self, rho_dots, lls_, min_ll_, margin=1.e-5):
        """
        # lls_ is the **SUM** of -2lnL from *ALL* burst sizes
        """
        ll_99 = 6.63
        rho_dots99 = np.interp(0.0, lls_-min_ll_-ll_99, rho_dots)
        if abs(np.interp(rho_dots99, rho_dots, lls_-min_ll_-ll_99)) <= margin:
            return rho_dots99, 0
        ul_99_idx = (np.abs(lls_-min_ll_-ll_99)).argmin()
        ul_99_idx_all = np.where(abs(lls_-lls_[ul_99_idx])<margin)
        if ul_99_idx_all[0].shape[0]==0:
            print("Can't find 99% UL!")
            return None
            #sys.exit(1)
        elif ul_99_idx_all[0].shape[0]>1:
            print("More than one 99% UL found, strange!")
            print("These are rho_dot = %s, and -2lnL = %s" % (rho_dots[ul_99_idx_all], lls_[ul_99_idx_all]))
            return rho_dots[ul_99_idx_all], lls_[ul_99_idx_all]
        else:
            print("99%% UL found at rho_dot = %.0f, and -2lnL = %.2f" % (rho_dots[ul_99_idx], lls_[ul_99_idx]))
            return rho_dots[ul_99_idx], lls_[ul_99_idx]

    """
    # won't work:
    def get_minimum_ll(self, burst_size, t_window, verbose=False):
        init_rho_dot = 2.e5
        results = minimize(self.get_ll, init_rho_dot, args=(burst_size, t_window), method='L-BFGS-B',bounds=[(0,1.e7)])
        if not results.success:
            print("Problem finding the minimum log likelihood!! ")
        minimum_rho_dot = results.x
        minimum_ll = results.fun
        if verbose:
            print("The minimum -2lnL is %.2f at rho_dot %.1f" % (minimum_ll, minimum_rho_dot) )
        return minimum_rho_dot, minimum_ll
    """

    def get_likelihood_dict(self):
        ll_dict={}
        residual_dict = self.residual_dict

        sig_bkg_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        #fill with zero if no burst size count
        for key_ in sig_bkg_burst_sizes:
            key_ = int(key_)
            if key_ not in self.sig_burst_hist:
                self.sig_burst_hist[key_] = 0
            if key_ not in self.avg_bkg_hist:
                self.avg_bkg_hist[key_] = 0
            residual_dict[key_] = self.sig_burst_hist[key_] - self.avg_bkg_hist[key_]
        return residual_dict


    def plot_burst_hist(self, filename=None, title="Burst histogram", plt_log=True, error="Poisson"):
        if not hasattr(self,'sig_burst_hist'):
            print("self.sig_burst_hist doesn't exist, what to plot?")
            return None
        if not hasattr(self,'avg_bkg_hist'):
            print("self.avg_bkg_hist doesn't exist, what to plot?")
            return None

        plt.figure(figsize=(10,8))
        ax1 = plt.subplot(3,1, (1,2))

        if self.avg_bkg_hist.keys() != self.sig_burst_hist.keys():
            for key in self.avg_bkg_hist.keys():
                if key not in self.sig_burst_hist:
                    self.sig_burst_hist[key]=0
            for key in self.sig_burst_hist.keys():
                if key not in self.avg_bkg_hist:
                    self.avg_bkg_hist[key]=0

        if error is None:
            sig_err = np.zeros(np.array(self.sig_burst_hist.values()).shape[0])
            bkg_err = np.zeros(np.array(self.avg_bkg_hist.values()).shape[0])
        elif error=="Poisson":
            sig_err = np.sqrt(np.array(self.sig_burst_hist.values()).astype('float64'))
            bkg_err = np.sqrt(np.array(self.avg_bkg_hist.values()).astype('float64'))
        elif error.lower()=="std":
            sig_err = np.sqrt(np.array(self.sig_burst_hist.values()).astype('float64'))
            all_bkg_burst_sizes = set(k for dic in self.bkg_burst_hists for k in dic.keys())
            bkg_err = np.zeros(sig_err.shape[0])
            for key_ in all_bkg_burst_sizes:
                key_ = float(key_)
                bkg_err[key_] = np.std(np.array([d[key_] for d in self.bkg_burst_hists if key_ in d]))

        ax1.errorbar(self.sig_burst_hist.keys()[1:], self.sig_burst_hist.values()[1:], xerr=0.5,
                         yerr=sig_err[1:], fmt='bs', capthick=0,
                         label="Data")
        ax1.errorbar(self.avg_bkg_hist.keys()[1:], self.avg_bkg_hist.values()[1:], xerr=0.5,
                         yerr=bkg_err[1:], fmt='rv', capthick=0,
                         label="Background")
        plt.title(title)
        ax1.set_ylabel("Counts")
        #plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        if plt_log:
            plt.yscale('log')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(loc='best')

        #plot residual
        residual_dict=self.get_residual_hist()

        if self.avg_bkg_hist.keys() != self.sig_burst_hist.keys():
            print("Check residual error calc")
        res_err = np.sqrt(sig_err**2+bkg_err**2)

        #plt.figure()
        ax2 = plt.subplot(3, 1, 3, sharex=ax1)
        ax2.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs', capthick=0,
                     label="Residual")
        ax2.axhline(y=0, color='gray', ls='--')
        ax2.set_xlabel("Burst size")
        ax2.set_ylabel("Counts")
        #plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        #plt.yscale('log')
        plt.legend(loc='best')
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_theta2(self, theta2s=np.arange(0, 2, 0.01), psf_width=0.1, N=100, const=1, ax=None, ylog=True):
        const_ = np.ones(theta2s.shape[0]) * const
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(111)
        ax.plot(theta2s, self.psf_func(theta2s, psf_width, N=N), 'r--')
        ax.plot(theta2s, const_, 'b:')
        ax.plot(theta2s, self.psf_func(theta2s, psf_width) + const_, 'k-')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\theta^2$ (deg$^2$)')
        ax.set_ylabel("Count")
        return ax

    def plot_skymap(self, coords, Es, ELs, ax=None, color='r', fov_center=None, fov=1.75, fov_color='gray',
                    cent_coords=None, cent_marker='+', cent_ms=1.8, cent_mew=4.0, cent_radius=0.01, cent_color='b',
                    label=None, projection=None):
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            if projection is not None:
                ax = plt.subplot(111, projection="aitoff")
            else:
                ax = plt.subplot(111)
        ax.plot(coords[:, 0], coords[:, 1], color + '.')
        label_flag = False
        for coor, E_, EL_ in zip(coords, Es, ELs):
            r = self.get_psf(E_, EL_)
            if label_flag == False:
                #circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False, label=label)
                ellipse = Ellipse(coor, r / np.cos(deg2rad(coor[1])) * 2, r * 2, color=color, fill=False, label=label)
                label_flag = True
            else:
                #circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
                ellipse = Ellipse(coor, r / np.cos(deg2rad(coor[1])) * 2, r * 2, color=color, fill=False)
            #ax.add_patch(circ)
            ax.add_patch(ellipse)

        label_flag = False
        if fov is not None and fov_center is not None:
            #circ_fov = plt.Circle(fov_center, radius=fov, color=fov_color, fill=False)
            ellipse_fov = Ellipse(fov_center, fov / np.cos(deg2rad(fov_center[1])) * 2, fov * 2, color=fov_color, fill=False)
            #ax.add_patch(circ_fov)
            ax.add_patch(ellipse_fov)
            ax.set_xlim(fov_center[0] - fov * 1.1, fov_center[0] + fov * 1.1)
            ax.set_ylim(fov_center[1] - fov * 1.1, fov_center[1] + fov * 1.1)
        if cent_coords is not None:
            # circ_cent=plt.Circle(cent_coords, radius=cent_radius, color=cent_color, fill=False)
            # ax.add_patch(circ_cent)
            ax.plot(cent_coords[0], cent_coords[1], marker=cent_marker, ms=cent_ms, markeredgewidth=cent_mew,
                    color=cent_color)

        plt.legend(loc='best')
        ax.set_xlabel('RA')
        ax.set_ylabel("Dec")
        return ax


class Pbh_combined(Pbh):
    def __init__(self, window_size):
        super(Pbh_combined, self).__init__()
        # the cut on -2lnL for rho_dot that gives an observed burst size
        #!!!note how this is different from the ll_cut on the centroid in the super class
        self.delta_ll_cut = 6.63
        self.photon_df = None
        self.pbhs = []
        self.runNums = []
        self.sig_burst_hist = {}
        self.avg_bkg_hist = {}
        self.residual_dict = {}
        #this is important, keep track of the number of runs to average over
        self.n_runs = 0
        #total exposure in unit of year
        self.total_time_year = 0
        #self.effective_volume = 0.0
        self.effective_volumes = {}
        self.minimum_lls = {}
        self.rho_dot_ULs = {}
        #Make the class for a specific window size
        self.window_size = window_size
        #Some global parameters
        self.bkg_method = "scramble"
        self.rando_method = "avg"
        self.N_scramble = 10
        self.verbose = False
        self.burst_sizes_set = set()
        self.rho_dots = np.arange(0., 3.e5, 100)

    def change_window_size(self, window_size):
        self.window_size = window_size
        #a bunch of stuff needs re-initialization
        orig_pbhs = self.pbhs
        self.pbhs = []
        self.sig_burst_hist = {}
        self.avg_bkg_hist = {}
        self.residual_dict = {}
        self.total_time_year = 0
        self.effective_volumes = {}
        self.minimum_lls = {}
        self.rho_dot_ULs = {}
        self.burst_sizes_set = set()
        #analyze again:
        if isinstance(orig_pbhs[0], Pbh_combined):
            for pbhs_ in orig_pbhs:
                pbhs_.window_size = window_size
                pbhs_.sig_burst_hist = {}
                pbhs_.avg_bkg_hist = {}
                pbhs_.residual_dict = {}
                pbhs_.total_time_year = 0
                pbhs_.effective_volumes = {}
                pbhs_.minimum_lls = {}
                pbhs_.rho_dot_ULs = {}
                pbhs_.burst_sizes_set = set()
                for pbh_ in pbhs_.pbhs:
                    _sig_burst_hist, _sig_burst_dict = pbh_.sig_burst_search(window_size=self.window_size, verbose=self.verbose)
                    _avg_bkg_hist, _bkg_burst_dicts = pbh_.estimate_bkg_burst(window_size=self.window_size, rando_method=self.rando_method,
                                                                       method=self.bkg_method,copy=True, n_scramble=self.N_scramble,
                                                                       return_burst_dict=True, verbose=self.verbose)
                    pbhs_.do_step2345(pbh_)
                self.do_step2345(pbhs_)
        else:
            for pbh_ in orig_pbhs:
                _sig_burst_hist, _sig_burst_dict = pbh_.sig_burst_search(window_size=self.window_size, verbose=self.verbose)
                _avg_bkg_hist, _bkg_burst_dicts = pbh_.estimate_bkg_burst(window_size=self.window_size, rando_method=self.rando_method,
                                                                   method=self.bkg_method,copy=True, n_scramble=self.N_scramble,
                                                                   return_burst_dict=True, verbose=self.verbose)
                self.do_step2345(pbh_)

        rho_dot_ULs = self.get_ULs()
        return rho_dot_ULs


    def add_pbh(self, pbh):
        #When adding a new run, we want to update:
        # 1. self.n_runs
        # 2. self.total_time_year
        # 3. self.sig_burst_hist
        # 4. self.avg_bkg_hist
        # 5. self.effective_volumes
        self.pbhs.append(pbh)
        # 1.
        previous_n_runs = self.n_runs
        self.n_runs += 1
        self.do_step2345(pbh)
        # in the combiner class this is the latest run Num
        self.runNum = pbh.runNum
        self.runNums.append(pbh.runNum)

    def do_step2345(self, pbh):
        # 2.
        if not hasattr(pbh, 'total_time_year'):
            pbh.total_time_year = (pbh.tOn*(1.-pbh.DeadTimeFracOn))/31536000.
        previous_total_time_year = self.total_time_year
        self.total_time_year += pbh.total_time_year
        # 3 and 4 and residual
        current_all_burst_sizes = self.get_all_burst_sizes()
        new_all_burst_sizes = current_all_burst_sizes.union(set(k for dic in [pbh.sig_burst_hist, pbh.avg_bkg_hist] for k in dic.keys()))
        self.burst_sizes_set = new_all_burst_sizes
        for key_ in new_all_burst_sizes:
            #first zero-pad the new pbh hists with all possible burst sizes
            key_ = int(key_)
            if key_ not in pbh.sig_burst_hist:
                pbh.sig_burst_hist[key_] = 0
            if key_ not in pbh.avg_bkg_hist:
                pbh.avg_bkg_hist[key_] = 0
        for key_ in new_all_burst_sizes:
            #now add all the bursts in pbh to self
            key_ = int(key_)
            if key_ not in self.sig_burst_hist:
                self.sig_burst_hist[key_] = pbh.sig_burst_hist[key_]
            else:
                self.sig_burst_hist[key_] += pbh.sig_burst_hist[key_]
            if key_ not in self.avg_bkg_hist:
                self.avg_bkg_hist[key_] = pbh.avg_bkg_hist[key_]
            else:
                self.avg_bkg_hist[key_] += pbh.avg_bkg_hist[key_]
            self.residual_dict[key_] = self.sig_burst_hist[key_] - self.avg_bkg_hist[key_]

        # 5.
        for key_ in new_all_burst_sizes:
            if key_ not in self.effective_volumes:
                #self.effective_volumes[key_] = pbh.total_time_year * pbh.V_eff(key_, self.window_size)
                #self.effective_volumes[key_] = self.V_eff(key_, self.window_size)
                self.effective_volumes[key_] = 0.0
                for pbh_ in self.pbhs:
                    #This already includes the new pbh object
                    Veff_ = pbh_.V_eff(key_, self.window_size, verbose=False)
                    self.effective_volumes[key_] += pbh_.total_time_year * Veff_ * 1.0
                new_total_time = 1.0 * (previous_total_time_year + pbh.total_time_year)
                self.effective_volumes[key_] = self.effective_volumes[key_]/new_total_time
            else:
                #self.effective_volumes[key_] already exists, so don't need to calculate again, this is faster
                new_total_time = 1.0 * (previous_total_time_year + pbh.total_time_year)
                self.effective_volumes[key_] = (self.effective_volumes[key_] * previous_total_time_year + pbh.total_time_year * pbh.V_eff(key_, self.window_size)) / new_total_time
            #if key_ not in pbh.effective_volumes:
            #    pbh.effective_volumes[key_] = pbh.V_eff(key_, self.window_size)

        assert self.burst_sizes_set==self.get_all_burst_sizes(), "Something is wrong when adding a pbh to pbh combined"

    #Override V_eff for the combiner class
    def V_eff(self, burst_size, t_window, verbose=False):
        assert t_window==self.window_size, "You are asking for an effective volume for a different window size."
        if burst_size not in self.effective_volumes:
            #recalculate effective volume from each single pbh class in the combiner
            self.effective_volumes[burst_size] = 0.0
            total_time = 0.0
            for pbh_ in self.pbhs:
                Veff_ = pbh_.V_eff(burst_size, self.window_size, verbose=False)
                self.effective_volumes[burst_size] += pbh_.total_time_year * Veff_ * 1.0
                total_time += pbh_.total_time_year
            self.effective_volumes[burst_size] = self.effective_volumes[burst_size]/total_time * 1.0
        return self.effective_volumes[burst_size]

    def get_all_burst_sizes(self):
        #returns a set, not a dict
        all_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        return all_burst_sizes

    def add_run(self, runNum):
        pbh_ = Pbh()
        pbh_.readEDfile(runNum=runNum)
        pbh_.get_TreeWithAllGamma(runNum=runNum, nlines=None)
        _sig_burst_hist, _sig_burst_dict = pbh_.sig_burst_search(window_size=self.window_size, verbose=self.verbose)
        _avg_bkg_hist, _bkg_burst_dicts = pbh_.estimate_bkg_burst(window_size=self.window_size, rando_method=self.rando_method,
                                                               method=self.bkg_method,copy=True, n_scramble=self.N_scramble,
                                                               return_burst_dict=True, verbose=self.verbose)
        pbh_.getRunSummary()
        self.add_pbh(pbh_)
        #self.runNums.append(runNum)

    def n_excess(self, rho_dot, Veff, verbose=False):
        #eq 8.8, or maybe more appropriately call it n_expected
        if not hasattr(self, 'total_time_year'):
            self.total_time_year = (self.tOn*(1.-self.DeadTimeFracOn))/31536000.
        #n_ex = 1.0 * rho_dot * self.total_time_year * Veff
        # because the burst likelihood cut -9.5 is at 90% CL
        n_ex = 0.9 * rho_dot * self.total_time_year * Veff
        if verbose:
            print("The value of the expected number of bursts (eq 8.9) is %.2f" % n_ex)
        return n_ex

    #Override get_ll so that it knows where to find the effective volume
    def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=None):
    #def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=100):
        #eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
        if upper_burst_size is None:
            all_burst_sizes = self.get_all_burst_sizes()
        else:
            all_burst_sizes = range(burst_size_threshold,upper_burst_size+1)
        ll_ = 0.0
        sum_nb = 0
        self.good_burst_sizes=[] # use this to keep burst sizes that are large enough so that $n_b > \sum_b n_{b+1}$
        for burst_size in np.sort(np.array(list(all_burst_sizes)))[::-1]:
        #for burst_size in all_burst_sizes:
            if burst_size >= burst_size_threshold:
                #Veff_ = self.V_eff(burst_size, t_window, verbose=verbose)
                #print("Burst size %d " % burst_size)
                #Veff_ = self.effective_volumes[burst_size]
                Veff_ = self.V_eff(burst_size, t_window, verbose=verbose)
                n_expected_ = self.n_excess(rho_dot, Veff_, verbose=verbose)
                if burst_size not in self.sig_burst_hist:
                    self.sig_burst_hist[burst_size] = 0
                if burst_size not in self.avg_bkg_hist:
                    self.avg_bkg_hist[burst_size] = 0
                n_on_ = self.sig_burst_hist[burst_size]
                n_off_ = self.avg_bkg_hist[burst_size]
                if n_on_ < sum_nb:
                    print("reached where n_b < \sum_b n_(b+1), at b={}".format(burst_size))
                    print("Stopping")
                    break
                else:
                    self.good_burst_sizes.append(burst_size)
                ll_ += self.ll(n_on_, n_off_, n_expected_)
                sum_nb += n_on_
                if verbose:
                    #print("###############################################################################")
                    print("Adding -2lnL at burst size %d, for search window %.1f and rate density %.1f, so far -2lnL = %.2f" % (burst_size, t_window, rho_dot, ll_))
                    #print("###############################################################################")
        if verbose:
            print("###############################################################################")
            print("-2lnL above burst size %d, for search window %.1f and rate density %.1f is %.2f" % (burst_size_threshold, t_window, rho_dot, ll_))
            print("###############################################################################")
        return ll_



    #@autojit
    def get_minimum_ll(self, burst_size, t_window, rho_dots=np.arange(0., 3.e5, 100), return_arrays=True,
                       #verbose=False, upper_burst_size=100):
                       verbose=False, upper_burst_size=None):
        #search rho_dots for the minimum -2lnL for burst size >= burst_size
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        min_ll_ = 1.e5
        rho_dot_min_ll_ = -1.0
        if return_arrays:
            lls_ = np.zeros(rho_dots.shape[0]).astype('float64')
        i=0
        for rho_dot_ in rho_dots:
            ll_ = self.get_ll(rho_dot_, burst_size, t_window, verbose=verbose, upper_burst_size=upper_burst_size)
            if ll_ < min_ll_:
                min_ll_ = ll_
                rho_dot_min_ll_ = rho_dot_
            if return_arrays:
                lls_[i] = ll_
                i += 1
        if return_arrays:
            return rho_dot_min_ll_, min_ll_, rho_dots, lls_
        return rho_dot_min_ll_, min_ll_

    def get_ULs(self, burst_size_threshold=2, rho_dots=None, upper_burst_size=None):
        print("Getting UL for burst size above %d..." % burst_size_threshold)
        if rho_dots is None:
            rho_dots = self.rho_dots
        minimum_rho_dot, minimum_ll, rho_dots, lls = self.get_minimum_ll(burst_size_threshold, self.window_size,
                                                                         rho_dots=rho_dots, verbose=self.verbose,
                                                                         upper_burst_size=upper_burst_size)
        self.minimum_lls[burst_size_threshold] = minimum_ll
        self.rho_dot_ULs[burst_size_threshold], ll_UL_ = self.get_ul_rho_dot(rho_dots, lls, minimum_ll, margin=1.e-5)
        return self.rho_dot_ULs

    def plot_ll_vs_rho_dots(self, save_hist="ll_vs_rho_dots", xlog=True, grid=True, plot_hline=True, show=False, ylim=(0,25)):
        rho_dots=self.rho_dots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for b_ in self.burst_sizes_set:
            if b_==1:
                continue
            minimum_rho_dot, minimum_ll, rho_dots, lls = self.get_minimum_ll(b_, self.window_size, rho_dots=rho_dots, verbose=self.verbose)
            plt.plot(rho_dots, lls-minimum_ll, label="burst size "+str(b_)+", "+str(self.window_size)+"-s window")
        #plt.axvline(x=minimum_rho_dot, color="b", ls="--",
        #            label=("minimum -2lnL = %.2f at rho_dot = %.1f " % (minimum_ll, minimum_rho_dot)))
        if plot_hline:
            plt.axhline(y=6.63, color="r", ls='--')
        plt.xlabel(r"Rate density of PBH evaporation (pc$^{-3}$ yr$^{-1}$)")
        plt.ylabel(r"-2$\Delta$lnL")
        plt.legend(loc='best')
        plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if grid:
            plt.grid(b=True)
        filename=save_hist+"_"+str(self.n_runs)+"runs.png"
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        print("Done!")

    def process_run_list(self, filename="pbh_runlist.txt"):
        runlist = pd.read_csv(filename, header=None)
        runlist.columns = ["runNum"]
        self.runlist = runlist.runNum.values
        self.bad_runs = []
        for run_ in self.runlist:
            try:
                self.add_run(run_)
                print("Run %d processed." % run_)
            except:
                print("*** Bad run: %d ***" % run_)
                #raise
                self.bad_runs.append(run_)
                self.runlist = self.runlist[np.where(self.runlist!=run_)]
        rho_dot_ULs = self.get_ULs()
        return rho_dot_ULs

    def save(self, filename):
        #save_hdf5(self, filename)
        dump_pickle(self, filename)

class powerlaw:
    # Class can calculate a power-law pdf, cdf from x_min to x_max,
    # always normalized so that the integral pdf is 1
    # (so is the diff in cdf between x_max and xmin)
    # ###########################################################################
    #                       Initialization of the object.                      #
    ############################################################################
    def __init__(self, a, x_min, x_max):
        """
        :param a: photon index, dN/dE = E^a
        :param x_min: e lo
        :param x_max: e hi
        :return:
        """
        self.a = a
        self.x_min = x_min
        self.x_max = x_max

    def pdf(self, x):
        if self.a == -1:
            return -1
        self.pdf_norm = (self.a + 1) / (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return self.pdf_norm * x ** self.a

    def cdf(self, x):
        if self.a == -1:
            return -1
        self.cdf_norm = 1. / (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return self.cdf_norm * (x ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))

    def ppf(self, q):
        if self.a == -1:
            return -1
        norm = (self.x_max ** (self.a + 1.0) - self.x_min ** (self.a + 1.0))
        return (q * norm * 1.0 + self.x_min ** (self.a + 1.0)) ** (1.0 / (self.a + 1.0))

    def random(self, n):
        r_uniform = np.random.random_sample(n)
        return self.ppf(r_uniform)

def plot_pbh_ll_vs_rho_dots(pbhs_list, rho_dots=np.arange(0., 3.e5, 100), burst_size_thresh=2, filename="ll_vs_rho_dots.png",
                            label_names=None, xlog=True, grid=True, plot_hline=True, show=False, xlim=None, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if label_names is not None:
        assert len(pbhs_list)==len(label_names), "Check the length of label_names, doesn't match pbhs_list"
    for i, p in enumerate(pbhs_list):
        if label_names is not None:
            label_name =  label_names[i]+" burst size "+str(burst_size_thresh)+", "+str(p.window_size)+"-s window"
        else:
            label_name =  "burst size "+str(burst_size_thresh)+", "+str(p.window_size)+"-s window"
        minimum_rho_dot, minimum_ll, rho_dots, lls = p.get_minimum_ll(burst_size_thresh, p.window_size, rho_dots=rho_dots, verbose=p.verbose)
        plt.plot(rho_dots, lls-minimum_ll, label=label_name)
    #plt.axvline(x=minimum_rho_dot, color="b", ls="--",
    #            label=("minimum -2lnL = %.2f at rho_dot = %.1f " % (minimum_ll, minimum_rho_dot)))
    if plot_hline:
        plt.axhline(y=6.63, color="r", ls='--')
    plt.xlabel(r"Rate density of PBH evaporation (pc$^{-3}$ yr$^{-1}$)")
    plt.ylabel(r"-2$\Delta$lnL")
    plt.legend(loc='best')
    if xlog:
        plt.xscale('log')
    if grid:
        plt.grid(b=True)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    print("Done!")


def test_psf_func(Nburst=10, filename=None, cent_ms=8.0, cent_mew=1.8):
    # Nburst: Burst size to visualize
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    EL = 15
    pl_nu = powerlaw(index, E_min, E_max)
    rand_Es = pl_nu.random(Nburst)
    rand_bkg_coords = np.zeros((Nburst, 2))
    rand_sig_coords = np.zeros((Nburst, 2))
    psfs = np.zeros(Nburst)

    for i in range(Nburst):
        psf_width = pbh.get_psf(rand_Es[i], EL)
        psfs[i] = psf_width
        rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
        rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
        rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
        rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

    cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
    cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)

    ax = pbh.plot_skymap(rand_bkg_coords, rand_Es, [EL] * Nburst, color='b', fov_center=fov_center,
                         cent_coords=cent_bkg, cent_marker='+', cent_ms=cent_ms, cent_mew=cent_mew,
                         label=("bkg ll=%.2f" % ll_bkg))
    pbh.plot_skymap(rand_sig_coords, rand_Es, [EL] * Nburst, color='r', fov_center=fov_center, ax=ax,
                    cent_coords=cent_sig, cent_ms=cent_ms, cent_mew=cent_mew,
                    label=("sig ll=%.2f" % ll_sig))
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return pbh


# def test_psf_func_sim(psf_width=0.05, prob="uniform", Nsim=10000, Nbins=40, filename=None, xlim=None):
def test_psf_func_sim(psf_width=0.05, prob="psf", Nsim=10000, Nbins=40, filename=None, xlim=(0, 0.5)):
    #def test_psf_func_sim(psf_width=0.05, prob="psf", Nsim=10000, Nbins=40, filename=None, xlim=None):
    pbh = Pbh()
    fov = 1.75

    # to store the value of a sim signal!
    rand_thetas = []
    for i in range(Nsim):
        rand_thetas.append(pbh.gen_one_random_theta(psf_width, prob=prob, fov=fov))

    rand_theta2s = np.array(rand_thetas)
    rand_theta2s = rand_theta2s ** 2

    theta2s = np.arange(0, fov, 0.01) ** 2

    theta2_hist, theta2_bins, _ = plt.hist(rand_theta2s, bins=Nbins, alpha=0.3, label="Monte Carlo")
    theta2s_analytical = pbh.psf_func(theta2s, psf_width, N=1)

    plt.yscale('log')
    plt.plot(theta2s, theta2s_analytical / theta2s_analytical[0] * theta2_hist[0], 'r--',
             label="Hyperbolic secant function")
    plt.xlim(xlim)
    plt.xlabel(r'$\theta^2$ (deg$^2$)')
    plt.ylabel("Count")
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def calc_cut_sig(ll_sig_all, ll_bkg_all, ll_cut, upper=True):
    ll_sig_all = np.array(ll_sig_all)
    ll_bkg_all = np.array(ll_bkg_all)
    if upper:
        s = float((ll_sig_all <= ll_cut).sum())
        b = float((ll_bkg_all <= ll_cut).sum())
    else:
        s = float((ll_sig_all >= ll_cut).sum())
        b = float((ll_bkg_all >= ll_cut).sum())
    sig = 1.0 * s / np.sqrt(s + b)
    return sig


def cut_optimize(ll_sig_all, ll_bkg_all, ll_cuts, upper=True, plot=True, outfile=None,
                 label="Burst size 3", sig_bins=50, bkg_bins=100, ylog=True):
    sigs = np.zeros_like(ll_cuts).astype('float')
    for i, c in enumerate(ll_cuts):
        sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
        sigs[i] = sig
    if plot:
        plt.plot(ll_cuts, sigs, color='g', label=str(label) + " SNR")
        plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label=str(label) + " signal")
        plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label=str(label) + " background")
        best_cut = ll_cuts[np.where(sigs == np.max(np.nan_to_num(sigs)))]
        plt.axvline(x=best_cut[0], ls="--", lw=0.3, label="Best cut {:.2f}".format(best_cut[0]))
        plt.axhline(y=sigs[np.where(sigs == np.max(np.nan_to_num(sigs)))][0], ls="--", lw=0.3)

        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        if ylog:
            plt.yscale('log')
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()
    return (sigs, best_cut[0])


def cut_90efficiency(ll_sig_all, ll_bkg_all, ll_cuts, upper=True, plot=True, outfile=None,
                     label="Burst size 3", sig_bins=50, bkg_bins=100, ylog=True):
    # for i, c in enumerate(ll_cuts):
    #    sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
    #    sigs[i] = sig
    if plot:
        # plt.plot(ll_cuts, sigs, color='g', label=str(label) + " SNR")
        plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label=str(label) + " signal")
        plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label=str(label) + " background")
        # ll_sig_all = np.nan_to_num(ll_sig_all)
        best_cut = np.percentile(ll_sig_all, 90)
        plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut {:.2f}".format(best_cut))
        # plt.axhline(y=sigs[np.where(ll_sig_all==best_cut)][0], ls="--", lw=0.3)

        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        if ylog:
            plt.yscale('log')
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()
    return best_cut


def sim_psf_likelihood(Nsim=1000, N_burst=3, filename=None,
                       sig_bins=50, bkg_bins=100, ylog=True,
                       EL=75, fov_center=np.array([180., 30.0])):
    pbh = Pbh()

    fov = 1.75

    # spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    # Burst size to visualize
    # N_burst = 10
    # EL = 15
    pl_nu = powerlaw(index, E_min, E_max)

    # Nsim = 1000
    ll_bkg_all = np.zeros(Nsim)
    ll_sig_all = np.zeros(Nsim)

    for j in range(Nsim):
        rand_Es = pl_nu.random(N_burst)
        rand_bkg_coords = np.zeros((N_burst, 2))
        rand_sig_coords = np.zeros((N_burst, 2))
        psfs = np.zeros(N_burst)

        for i in range(N_burst):
            psf_width = pbh.get_psf(rand_Es[i], EL)
            psfs[i] = psf_width
            rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
            rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
            rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
            rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

        cent_bkg, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)
        cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)
        ll_bkg_all[j] = ll_bkg
        ll_sig_all[j] = ll_sig
    return ll_sig_all, ll_bkg_all


def test_sim_likelihood(Nsim=1000, N_burst=3, filename=None, sig_bins=50, bkg_bins=100, ylog=True):
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=filename, sig_bins=sig_bins, bkg_bins=bkg_bins, ylog=ylog)
    plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label="Burst size " + str(N_burst) + " signal")
    plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label="Burst size " + str(N_burst) + " background")
    plt.axvline(x=-9.5, ls="--", lw=0.3)
    plt.legend(loc='best')
    plt.xlabel("Likelihood")
    if ylog:
        plt.yscale('log')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None


def sim_psf_likelihood_scramble_data(Nsim=1000, N_burst=3,
                       runNum=55480,
                       filename=None,
                       sig_bins=50, bkg_bins=100, ylog=True,
                       ):
    #EL = 75, fov_center = np.array([180., 30.0])
    fov=1.75
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=None)


    if Nsim >= pbh.photon_df.shape[0] - 1:
        print("Only {} events, doing {} sims instead of {}...".format(pbh.photon_df.shape[0], pbh.photon_df.shape[0] - 1, Nsim))
        Nsim = pbh.photon_df.shape[0] - 1
    # Nsim = 1000
    ll_bkg_all = np.zeros(Nsim)
    ll_sig_all = np.zeros(Nsim)

    for j in range(Nsim):
        pbh.scramble()

        #
        #this_slice = pbh.photon_df.iloc[j*N_burst:(j+1)*N_burst]
        this_slice = pbh.photon_df.iloc[j:j+N_burst]
        rand_Es = this_slice.Es.values

        #rand_bkg_coords = np.zeros((N_burst, 2))
        rand_bkg_coords = np.array([this_slice.RAs, this_slice.Decs]).T
        rand_sig_coords = np.zeros((N_burst, 2))
        psfs = this_slice.psfs.values

        fov_center, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)

        for i in range(N_burst):
            psf_width = psfs[i]
            rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
            rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

        cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)
        ll_bkg_all[j] = ll_bkg
        ll_sig_all[j] = ll_sig
    return ll_sig_all, ll_bkg_all


def test_sim_likelihood_from_data(Nsim=1000, N_burst=3, runNum=55480, filename=None, sig_bins=50, bkg_bins=100, ylog=True):
    ll_sig_all, ll_bkg_all = sim_psf_likelihood_scramble_data(Nsim=Nsim, N_burst=N_burst, runNum=runNum)
    best_cut = np.percentile(ll_sig_all, 90)
    plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label="Burst size " + str(N_burst) + " signal")
    plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label="Burst size " + str(N_burst) + " background")
    plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut = {:.2f}".format(best_cut))
    plt.legend(loc='best')
    plt.xlabel("Likelihood")
    if ylog:
        plt.yscale('log')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None


def test_sim_likelihood_from_data_all(Nsim=1000, N_bursts=range(2,11), runNum=55480, filename=None, sig_bins=50, bkg_bins=100, ylog=True):
    fov=1.75
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=None)


    #if Nsim >= pbh.photon_df.shape[0] - 1:
    #    print("Only {} events, doing {} sims instead of {}...".format(pbh.photon_df.shape[0], pbh.photon_df.shape[0] - 1, Nsim))
    #    Nsim = pbh.photon_df.shape[0] - 1
    # Nsim = 1000
    ll_bkg_all = np.zeros((len(N_bursts), Nsim)).astype(float)
    ll_sig_all = np.zeros((len(N_bursts), Nsim)).astype(float)
    best_cuts = np.zeros(len(N_bursts))
    fig, axes = plt.subplots(3, len(N_bursts)/3, figsize=(18, 18))

    for xx, N_burst in enumerate(N_bursts):
        sim_counter = 0
        N_evt_segments = pbh.photon_df.shape[0]//N_burst
        while sim_counter < Nsim:
            pbh.scramble()
            for j in range(N_evt_segments):
                #pbh.scramble()
                #
                this_slice = pbh.photon_df.iloc[j*N_burst:(j+1)*N_burst]
                #this_slice = pbh.photon_df.iloc[j:j+N_burst]
                #rand_Es = this_slice.Es.values

                #rand_bkg_coords = np.zeros((N_burst, 2))
                rand_bkg_coords = np.array([this_slice.RAs, this_slice.Decs]).T
                rand_sig_coords = np.zeros((N_burst, 2))
                psfs = this_slice.psfs.values

                fov_center, ll_bkg = pbh.minimize_centroid_ll(rand_bkg_coords, psfs)

                for i in range(N_burst):
                    psf_width = psfs[i]
                    rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
                    rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

                cent_sig, ll_sig = pbh.minimize_centroid_ll(rand_sig_coords, psfs)
                ll_bkg_all[xx, sim_counter] = ll_bkg
                ll_sig_all[xx, sim_counter] = ll_sig
                sim_counter += 1
                if sim_counter >= Nsim:
                    break

        best_cuts[xx] = np.percentile(ll_sig_all[xx], 90)
        axes.flatten()[xx].hist(ll_sig_all[xx], bins=sig_bins, color='r', alpha=0.3, label="Burst size " + str(N_burst) + " signal")
        axes.flatten()[xx].hist(ll_bkg_all[xx], bins=bkg_bins, color='b', alpha=0.3, label="Burst size " + str(N_burst) + " background")
        axes.flatten()[xx].axvline(x=best_cuts[xx], ls="--", lw=0.3, label="90% efficiency cut = {:.2f}".format(best_cuts[xx]))
        axes.flatten()[xx].legend(loc='best')
        axes.flatten()[xx].set_xlabel("Likelihood")
        if ylog:
            axes.flatten()[xx].set_yscale('log')
    plt.tight_layout()
    print(best_cuts)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None



def opt_cut(Nsim=10000, N_burst=3):
    #optimize cuts based on Monte Carlo, maximizing significance
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None, sig_bins=50, bkg_bins=100, ylog=True)
    ll_cuts=np.arange(-15,40,0.1)
    sigs, best_cut = cut_optimize(ll_sig_all, ll_bkg_all, ll_cuts,
                                  label="Burst size "+str(N_burst),
                                  outfile="psf_likelihood_cut_optimization_sim"+str(Nsim)+"_burst_size"+str(N_burst)+".png")
    return best_cut

def eff_cut(Nsim=10000, N_burst=3):
    #find 90% efficiency cuts
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None, sig_bins=50, bkg_bins=100, ylog=True)
    ll_cuts=np.arange(-15,40,0.1)
    best_cut = cut_90efficiency(ll_sig_all, ll_bkg_all, ll_cuts,
                                  label="Burst size "+str(N_burst),
                                  outfile="psf_likelihood_cut_90efficiency_sim"+str(Nsim)+"_burst_size"+str(N_burst)+".png")
    return best_cut


def sim_cut_90efficiency(NMC=50, Nsim=2000, N_burst=3, upper=True, plot=True, outfile=None,
                         EL = 75, fov_center = np.array([180., 30.0]),
                         cut_bins=50, ylog=False):
    cuts = np.zeros(NMC).astype('float')
    for trial in range(NMC):
        ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None,
                                                    EL = EL, fov_center = fov_center,
                                                    sig_bins=50, bkg_bins=100, ylog=True)
        ll_cuts=np.arange(-15,40,0.05)
        label="Burst size "+str(N_burst) + ", Dec {:.0f}".format(fov_center[1])
        #for i, c in enumerate(ll_cuts):
        #    sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
        #    sigs[i] = sig
        best_cut = np.percentile(ll_sig_all, 90)
        cuts[trial] = best_cut
    if plot:
        plt.hist(cuts, bins=cut_bins, color='r', alpha=0.3, label=str(label) + " sim cuts")
        #plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut {:.2f}".format(best_cut))
        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        if ylog:
            plt.yscale('log')
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()
    return cuts


def sim_cut_90efficiency_from_data(NMC=10, Nsim=1000, N_burst=3, plot=True, outfile=None,
                         runNum=55480,
                         cut_bins=50, ylog=False):
    cuts = np.zeros(NMC).astype('float')
    for trial in range(NMC):
        ll_sig_all, ll_bkg_all = sim_psf_likelihood_scramble_data(Nsim=Nsim, N_burst=N_burst, runNum=runNum)
        ll_cuts=np.arange(-15,40,0.05)
        label="Burst size "+str(N_burst) + ", Run {}".format(runNum)
        #for i, c in enumerate(ll_cuts):
        #    sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
        #    sigs[i] = sig
        best_cut = np.percentile(ll_sig_all, 90)
        cuts[trial] = best_cut
    if plot:
        plt.hist(cuts, bins=cut_bins, color='r', alpha=0.3, label=str(label) + " sim cuts")
        #plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut {:.2f}".format(best_cut))
        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        if ylog:
            plt.yscale('log')
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()
    return cuts



def test_burst_finding(window_size=3, runNum=55480, nlines=None, N_scramble=3, plt_log=True, verbose=False,
                       save_hist="test_burst_finding_histo", bkg_method="scramble", rando_method="avg"):
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=nlines)
    #do a small list
    #pbh.photon_df = pbh.photon_df[:nlines]
    sig_burst_hist, sig_burst_dict = pbh.sig_burst_search(window_size=window_size, verbose=verbose)

    #avg_bkg_hist = pbh.estimate_bkg_burst(window_size=window_size, method="scramble", copy=True, n_scramble=N_scramble)
    avg_bkg_hist, bkg_burst_dicts = pbh.estimate_bkg_burst(window_size=window_size, method=bkg_method, rando_method=rando_method,
                                                           copy=True, n_scramble=N_scramble, return_burst_dict=True, verbose=verbose)

    dump_pickle(sig_burst_hist, save_hist+str(window_size)+"_sig_hist.pkl")
    dump_pickle(sig_burst_dict, save_hist+str(window_size)+"_sig_dict.pkl")
    dump_pickle(bkg_burst_dicts, save_hist+str(window_size)+"_bkg_dicts.pkl")

    if nlines is None:
        filename=save_hist+"_AllEvts"+"_Nscrambles"+str(N_scramble)+"_window"+str(window_size)+"_method_"+str(bkg_method)+".png"
    else:
        filename=save_hist+"_Nevts"+str(nlines)+"_Nscrambles"+str(N_scramble)+"_window"+str(window_size)+"_method_"+str(bkg_method)+".png"

    pbh.plot_burst_hist(filename=filename, title="Burst histogram "+str(window_size)+"-s window "+str(bkg_method)+" method",
                        plt_log=True, error="Poisson")

    print("Done!")

    return pbh


def test_ll(window_sizes=[1,2,5,10], colors=['k', 'r', 'b', 'magenta'], runNum=55480, N_scramble=3, verbose=False,
            rho_dots=np.arange(0., 5.e5, 100), save_hist="test_ll", bkg_method="scramble", rando_method="avg",
            burst_size=2, xlog=True, grid=True):
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=None)

    for ii, window_size in enumerate(window_sizes):
        sig_burst_hist, sig_burst_dict = pbh.sig_burst_search(window_size=window_size, verbose=verbose)
        avg_bkg_hist, bkg_burst_dicts = pbh.estimate_bkg_burst(window_size=window_size, method=bkg_method, rando_method=rando_method,
                                                               copy=True, n_scramble=N_scramble, return_burst_dict=True, verbose=verbose)
        #rho_dots, lls = pbh.get_ll_vs_rho_dot(burst_size, window_size, rho_dots=rho_dots, verbose=verbose)
        #minimum_rho_dot, minimum_ll = pbh.get_minimum_ll(burst_size, window_size, verbose=verbose, return_arrays=False)
        minimum_rho_dot, minimum_ll, rho_dots, lls = pbh.get_minimum_ll(burst_size, window_size, rho_dots=rho_dots, verbose=verbose)
        plt.plot(rho_dots, lls-minimum_ll, color=colors[ii],label="burst size "+str(burst_size)+", "+str(window_size)+"-s window")
    #plt.axvline(x=minimum_rho_dot, color="b", ls="--",
    #            label=("minimum -2lnL = %.2f at rho_dot = %.1f " % (minimum_ll, minimum_rho_dot)))
    plt.axhline(y=6.63, color="r", ls='--')
    plt.xlabel(r"Rate density of PBH evaporation (pc$^{-3}$ yr$^{-1}$)")
    plt.ylabel(r"-2$\Delta$lnL")
    plt.legend(loc='best')
    if xlog:
        plt.xscale('log')
    if grid:
        plt.grid(b=True)
    filename=save_hist+"_AllEvts.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print("Done!")

    return pbh


def load_hdf5(fname):
    with tables.open_file(fname) as h5f:
        data = h5f.root.image[:]
    return data


def save_hdf5(data,ofname,compress=False, complevel=5, complib='zlib'):
    with tables.open_file(ofname,'w') as h5f:
        atom = tables.Atom.from_dtype(data.dtype)
        shape = data.shape
        if compress:
            filters = tables.Filters(complevel=complevel, complib=complib)
            ca = h5f.create_carray(h5f.root, 'image', atom, shape,filters=filters)
        else:
            ca = h5f.create_carray(h5f.root, 'image', atom, shape)
        ca[:] = data[:]


def load_pickle(f):
    inputfile = open(f, 'rb')
    loaded = pickle.load(inputfile)
    inputfile.close()
    return loaded


def dump_pickle(a, f):
    output = open(f, 'wb')
    pickle.dump(a, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()


def test1():
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    ras = np.random.random(size=10) * 2.0 + fov_center[0]
    decs = np.random.random(size=10) * 1.5 + fov_center[1]
    coords = np.concatenate([ras.reshape(10, 1), decs.reshape(10, 1)], axis=1)
    psfs = np.ones(10) * 0.1
    centroid = pbh.minimize_centroid_ll(coords, psfs)

    print(centroid)
    print(centroid.reshape(1, 2)[:, 0], centroid.reshape(1, 2)[:, 1])

    ax = pbh.plot_skymap(coords, [0.1] * 10, [0.2] * 10)
    pbh.plot_skymap(centroid.reshape(1, 2), [0.1], [0.2], ax=ax, color='b', fov_center=fov_center)
    plt.show()


def test_singlet_remover(Nburst=10, filename=None, cent_ms=8.0, cent_mew=1.8):
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    EL = 75
    pl_nu = powerlaw(index, E_min, E_max)
    rand_Es = pl_nu.random(Nburst)
    rand_bkg_coords = np.zeros((Nburst, 2))
    rand_sig_coords = np.zeros((Nburst, 2))
    psfs = np.zeros(Nburst)

    for i in range(Nburst):
        psf_width = pbh.get_psf(rand_Es[i], EL)
        psfs[i] = psf_width
        rand_bkg_theta = pbh.gen_one_random_theta(psf_width, prob="uniform", fov=fov)
        rand_sig_theta = pbh.gen_one_random_theta(psf_width, prob="psf", fov=fov)
        rand_bkg_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_bkg_theta)
        rand_sig_coords[i, :] = pbh.gen_one_random_coords(fov_center, rand_sig_theta)

    pbh.read_photon_list(np.arange(10), rand_bkg_coords[:, 0], rand_bkg_coords[:, 1], rand_Es, np.ones(10) * EL)
    slice = np.arange(10)
    slice, singlet_slice = pbh.singlet_remover(slice)
    print(slice)

def plot_Veff(pbh, window_sizes=[1, 10, 100], burst_sizes=range(2,11), lss=['-', '--', ':'], cs=['r', 'b', 'k'],
              draw_grid=True, filename="Effective_volume.png"):
    for i, window_ in enumerate(window_sizes):
        Veffs=[]
        for b in burst_sizes:
            Veffs.append(pbh.V_eff(b, window_))
        plt.plot(burst_sizes, Veffs, color=cs[i], ls=lss[i], label=("search window %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    plt.yscale('log')
    plt.xlabel("burst size")
    plt.ylabel(r"effective volume (pc$^3$)")
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_Veff_step(pbh, window_sizes=[1, 10, 100], burst_sizes=range(2,11), lss=['-', ':', '--'], cs=['r', 'b', 'k'],
              draw_grid=True, filename="Effective_volume.png"):
    for i, window_ in enumerate(window_sizes):
        Veffs=[]
        for b in burst_sizes:
            Veffs.append(pbh.V_eff(b, window_))
        plt.step(burst_sizes, Veffs, color=cs[i], where='mid', linestyle=lss[i], label=(r"$\Delta$t = %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    plt.yscale('log')
    plt.xlabel("Burst size")
    plt.ylabel(r"Effective volume (pc$^3$)")
    plt.legend(loc='best')
    plt.ylim(1e-4,1.1)
    if filename is not None:
        plt.savefig(filename)
    plt.show()



def plot_residual_vs_n_expected(pbhs, rho_dots, colors=None, draw_grid=True, ylim=None,
                                filename="residual_vs_n_expected.png", show=True, ylog=False):
    n_expected=np.zeros(len(pbhs.burst_sizes_set))
    if not isinstance(rho_dots, list):
        rho_dots = [rho_dots]
    if colors is not None:
        if len(colors) != len(rho_dots):
            print("colors provided has a different length from rho_dots!")
            colors=None
    residual_dict=pbhs.get_residual_hist()
    sig_err = np.sqrt(np.array(pbhs.sig_burst_hist.values()).astype('float64'))
    bkg_err = np.sqrt(np.array(pbhs.avg_bkg_hist.values()).astype('float64'))
    res_err = np.sqrt(sig_err**2+bkg_err**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs', capthick=0,
                 label="Residual")
    for k, rho_dot in enumerate(rho_dots):
        for i, b in enumerate(pbhs.burst_sizes_set):
            n_expected[i]=pbhs.n_excess(rho_dot, pbhs.effective_volumes[b])
        if colors is not None:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], color=colors[k], label=r"Expected number of bursts $\dot{\rho}$="+str(rho_dot))
        else:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], label=r"Expected number of bursts $\dot{\rho}$="+str(rho_dot))
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel("Burst size")
    ax.set_ylabel("Counts")
    if ylim is not None:
        plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
    plt.legend(loc='best')
    if draw_grid:
        plt.grid(b=True)
    #plt.yscale('log')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()

def plot_residual_UL_n_expected(pbhs, rho_dots, ULs, colors=None, draw_grid=True, ylim=None,
                                filename="residual_UL_n_expected.png", show=True, ylog=False, verbose=True):
    n_expected=np.zeros(len(pbhs.burst_sizes_set))
    if not isinstance(rho_dots, list):
        rho_dots = [rho_dots]
    if colors is not None:
        if len(colors) != len(rho_dots):
            print("colors provided has a different length from rho_dots!")
            colors=None
    residual_dict=pbhs.get_residual_hist()
    sig_err = np.sqrt(np.array(pbhs.sig_burst_hist.values()).astype('float64'))
    bkg_err = np.sqrt(np.array(pbhs.avg_bkg_hist.values()).astype('float64'))
    res_err = np.sqrt(sig_err**2+bkg_err**2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs', capthick=0,
                 label="Residual")
    for k, rho_dot in enumerate(rho_dots):
        for i, b in enumerate(pbhs.burst_sizes_set):
            n_expected[i]=pbhs.n_excess(rho_dot, pbhs.effective_volumes[b])
        if colors is not None:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], color=colors[k], label=r"Expected excess of bursts $\dot{\rho}$="+str(rho_dot))
        else:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], label=r"Expected excess of bursts $\dot{\rho}$="+str(rho_dot))
        if verbose:
            print("Expected rate for rho_dot=%.1f is" % rho_dot)
            print(n_expected)

    ax.plot(residual_dict.keys()[1:], ULs, color='r', marker='v', label="90% UL Helene")
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel("Burst size")
    ax.set_ylabel("Counts")
    if ylim is not None:
        plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
    plt.legend(loc='best')
    if draw_grid:
        plt.grid(b=True)
    #plt.yscale('log')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    #return n_expected

def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=55480, nlines=1000)
    print(pbh.photon_df.head())
    return pbh


def get_n_expected(pbh, t_window, r, verbose=False):
    #get expected gamma-ray rate from a pbh signal at distance r (pc)
    #eq 8.3; no acceptance, assume an on-axis event
    I_Value = pbh.get_integral_expected(pbh.kT_BH(t_window))
    I_Value = I_Value**(2./3.)/(4.*np.pi*r*r)
    if verbose:
        print("The expected gamma-ray rate from a pbh signal at distance %.4f (pc) is %.2f" % (r, I_Value))
    return I_Value

def plot_n_expected(pbh, t_window, rs=np.arange(1e-2,30,1e-2), ax=None, color='b', label=None):
    n_exps = np.zeros_like(rs)
    for i, r in enumerate(rs):
        n_exps[i] = get_n_expected(pbh, t_window, r)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('N_expected')
    ax.plot(rs, n_exps, color=color, label=label)
    return ax

def plot_n_expected3(pbh, t_window, rs=np.arange(1e-2,30,1e-2), ax=None, color='b', label=None):
    n_exps = np.zeros_like(rs)
    for i, r in enumerate(rs):
        n_exps[i] = get_n_expected(pbh, t_window, r)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Volume (pc$^3$)')
        ax.set_ylabel('N_expected')
    ax.plot(4./3.*np.pi*rs**3, n_exps, color=color, label=label)
    return ax

def plot_n_expected_all(pbh, t_windows, rs=np.arange(1e-2,30,1e-2), ax=None, colors=None, labels=None, show=True, save=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('N_expected')
    for i, t in enumerate(t_windows):
        if labels is not None:
            l = labels[i]
        else:
            l = None
        if colors is not None:
            plot_n_expected(pbh, t, rs=rs, ax=ax, color=colors[i], label=l)
        else:
            plot_n_expected(pbh, t, rs=rs, ax=ax, label=l)
    plt.legend(loc='best')
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()


def combine_from_pickle_list(listname, window_size, filetag="", burst_size_threshold=2, rho_dots=None, upper_burst_size=None):
    pbhs_combined_all_ = combine_pbhs_from_pickle_list(listname)
    print("Total exposure time is %.2f hrs" % (pbhs_combined_all_.total_time_year*365.25*24))
    pbhs_combined_all_.get_ULs(burst_size_threshold=burst_size_threshold, rho_dots=rho_dots, upper_burst_size=upper_burst_size)
    print("The effective volume above burst size 2 is %.6f pc^3" % (pbhs_combined_all_.effective_volumes[2]))
    print("There are %d runs in total" % (len(pbhs_combined_all_.runNums)))
    total_N_runs = len(pbhs_combined_all_.runNums)
    pbhs_combined_all_.plot_burst_hist(filename="burst_hists_test_"+str(filetag)+"_window"+str(window_size)+"-s_all"+str(total_N_runs)+"runs.png", title="Burst histogram "+str(window_size)+"-s window "+str(total_N_runs)+" runs", plt_log=True, error="Poisson")
    pbhs_combined_all_.plot_ll_vs_rho_dots(save_hist="ll_vs_rho_dots_test_"+str(filetag)+"_window"+str(window_size)+"-s_all"+str(total_N_runs)+"runs")
    return pbhs_combined_all_

def comp_pbhs(pbhs1, pbhs2):
    if pbhs1.total_time_year==pbhs2.total_time_year:
        print("Same total time")
    else:
        print("*** Different total time! ***")
    if pbhs1.effective_volumes==pbhs2.effective_volumes:
        print("Same effective volumes")
    else:
        print("*** Different effective volumes! ***")
    if pbhs1.get_all_burst_sizes()==pbhs2.get_all_burst_sizes():
        print("Same burst sizes")
    else:
        print("*** Different burst sizes! ***")
    if pbhs1.sig_burst_hist==pbhs2.sig_burst_hist:
        print("Same sig_burst_hist")
    else:
        print("*** Different sig_burst_hist! ***")
        print("{0}, {1}".format(pbhs1.sig_burst_hist, pbhs2.sig_burst_hist))
    if pbhs1.avg_bkg_hist==pbhs2.avg_bkg_hist:
        print("Same avg_bkg_hist")
    else:
        print("*** Different avg_bkg_hist! ***")
        print("{0}, {1}".format(pbhs1.avg_bkg_hist, pbhs2.avg_bkg_hist))
    if pbhs1.window_size==pbhs2.window_size:
        print("Same window_size")
    else:
        print("*** Different window_size! ***")
    if pbhs1.get_ULs()==pbhs2.get_ULs():
        print("Same UL for burst size threshold 2")
    else:
        print("*** Different ULs for burst size threshold 2! ***")


def process_one_run(run, window_size, rho_dots=np.arange(0., 3.e5, 100), plot=False, bkg_method="scramble"):
    pbhs = Pbh_combined(window_size)
    pbhs.rho_dots=rho_dots
    pbhs.bkg_method = bkg_method
    try:
        pbhs.add_run(run)
        print("Run %d processed." % run)
    except:
        print("*** Bad run: %d ***" % run)
        raise
    dump_pickle(pbhs, "pbhs_bkg_method_"+str(bkg_method)+"_run"+str(run)+"_window"+str(window_size)+"-s.pkl")
    if plot:
        pbhs.plot_ll_vs_rho_dots(save_hist="ll_vs_rho_dots_run"+str(run)+"_window"+str(window_size)+"-s")
        pbhs.plot_burst_hist(filename="burst_hists_run"+str(run)+"_window"+str(window_size)+"-s.png",
                             title="Burst histogram run"+str(run)+""+str(window_size)+"-s window ", plt_log=True, error="Poisson")

def ll(n_on, n_off, n_expected):
        #eq 8.13 without the sum
        return -2. * (-1. * n_expected + n_on * np.log(n_off + n_expected))

def calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0):
    #eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
    n_expected = 0.9 * rho_dot * total_time_year * Veff
    return ll(n_on, n_off, n_expected)

def sum_ll00(pbh, rho_dot, total_time_year=None, window_sizes=[1], burst_sizes=range(2,11), lss=['-', '--', ':'], cs=['r', 'b', 'k'],
              draw_grid=True, filename="ll.png", verbose=True):
    for i, window_ in enumerate(window_sizes):
        n_exps = np.zeros_like(burst_sizes).astype('float64')
        lls = np.zeros_like(burst_sizes).astype('float64')
        if total_time_year is None:
            total_time_year = pbh.total_time_year
        Veffs=[]
        for j, b in enumerate(burst_sizes):
            Veff = pbh.V_eff(b, window_)
            #print("total_time_year = %.5f, rho_dot = %.4f, Veff = %.8f" % \
            #      (total_time_year, rho_dot, Veff))
            n_expected = 0.9 * rho_dot * total_time_year * Veff
            n_exps[j] = n_expected
            #print("n_expected=%.10f ll=%.10f" % (n_expected, ll(0, 0, n_expected)))
            Veffs.append(Veff)
            lls[j] = float(calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0))
            #lls[j] = ll(0, 0, n_expected)
            if verbose:
                print("Burst size %d"%b)
                print("n_expected=%.10f ll=%.10f" % (n_expected, calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0)))
                print("log likelihood for burst size %d, 0 ON and 0 OFF is %.5f" % (b, lls[j]))
        plt.plot(burst_sizes, lls, color=cs[i], ls=lss[i], label=("search window %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("burst size")
    plt.ylabel(r"log likelihood (-2lnL)")
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()
    return lls, n_exps

def fit_gaussian_hist(bins, n):
    """ input is the bin edges and bin content returned by plt.hist. """
    def gaus(x, a, b, c):
        return a * np.exp(-(x - b)**2.0 / (2 * c**2))
    x = [0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)]
    y = n
    popt, pcov = curve_fit(gaus, x, y, p0=(-10, np.average(x, weights=n), 0.2))
    print("Fit results {}".format(popt))
    print("covariance matrix {}".format(pcov))

    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = gaus(x_fit, *popt)
    #returns x, y for plotting, and mean and sigma from fit
    return x_fit, y_fit, popt[1], popt[2], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2]


def psf_cut_search(NMC=1000, Nsim=1000, bss = range(2,11), fov_center = np.array([180., 80.0])):
    #cut_list = []
    best_cuts = []
    dec = fov_center[1]
    #for bs_ in range(2,10): #kernel died
    for bs_ in bss:
        sim_cuts_Dec80_ = sim_cut_90efficiency(NMC=NMC, Nsim=Nsim, N_burst=bs_,
                                          fov_center = fov_center,
                                          outfile="sim_cuts_distr_bs"+str(bs_)+"_Dec"+("{:.0f}".format(dec))+"_1M_sims_v4.png")

        np.save("sim_cuts_distr_bs"+str(bs_)+"_Dec"+("{:.0f}".format(dec))+"_1M_sims_v4.npy", sim_cuts_Dec80_)
        #cut_list.append(sim_cuts_Dec80_)
        hist_, bins_, _ = plt.hist(sim_cuts_Dec80_, bins=50, color='r', alpha=0.3, label="burst size {}, Dec={:.0f}$^\circ$".format(bs_, dec))
        x_fit, y_fit, mu, sig, dmu, dsig = fit_gaussian_hist(bins_, hist_)
        plt.plot(x_fit, y_fit, 'r--')
        #plt.axvline(mu, color='r', ls='--', label="mean: {:.2f}$\pm${:.2f} \n sigma: {:.2f}$\pm${:.2f}".format(mu, dmu, sig, dsig))
        plt.axvline(mu, color='r', ls='--', label="mean: {:.3f}$\pm${:.3f} \n sigma: {:.3f}$\pm${:.3f}".format(mu, dmu, sig, dsig))
        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        plt.savefig("sim_cuts_distr_bs"+str(bs_)+"_Dec"+("{:.0f}".format(dec))+"_1M_sims_fit_v4.pdf")
        best_cuts.append(mu)
        #plt.show()
    print("best cuts: {}")
    return best_cuts





def compare_run(runid, window_size=10, outfile=None, ylog=True, pkldir="batch_all_scramble_all_events", show=True):
    # compare burst histograms with Simon's
    if window_size==1:
        test_pbh = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window1.0-s.pkl")
        tp = test_pbh.pbhs[0]
    elif window_size==2:
        test_pbh2 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window2.0-s.pkl")
        tp = test_pbh2.pbhs[0]
    elif window_size==5:
        test_pbh5 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window5.0-s.pkl")
        tp = test_pbh5.pbhs[0]
    elif window_size==10:
        test_pbh10 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window10.0-s.pkl")
        tp = test_pbh10.pbhs[0]



    fname="/raid/reedbuck/archs/PBHAnalysis_cori/MoreFinalResults/run_"+str(runid)+"_burst.root"

    rf=ROOT.TFile(fname, "read")
    if window_size==1:
        h10b = rf.Get("1.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("1.0-second_run_" + str(runid) + "_Data");
    elif window_size == 2:
        h10b = rf.Get("2.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("2.0-second_run_" + str(runid) + "_Data");
    elif window_size == 5:
        h10b = rf.Get("5.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("5.0-second_run_" + str(runid) + "_Data");
    elif window_size == 10:
        h10b=rf.Get("10.0-second_run_"+str(runid)+"_Background");
        h10=rf.Get("10.0-second_run_"+str(runid)+"_Data");

    x=np.zeros(6)
    y=np.zeros(6)
    dy=np.zeros(6)

    xB=np.zeros(6)
    yB=np.zeros(6)
    dyB=np.zeros(6)

    for i in range(6):
        xB[i]=h10b.GetBinCenter(i+1)
        yB[i]=h10b.GetBinContent(i+1)
        dyB[i]=h10b.GetBinError(i+1)
        x[i] = h10.GetBinCenter(i + 1)
        y[i] = h10.GetBinContent(i + 1)
        dy[i] = h10.GetBinError(i + 1)


    plt.errorbar(x, y, xerr=0.5, yerr=dy, label="Simon's signal",
                 fmt='v', color='r', ecolor='r', capthick=0)
    plt.errorbar(xB, yB, xerr=0.5, yerr=dyB, label="Simon's background",
                 fmt='<', color='b', ecolor='b', capthick=0)
    plt.errorbar(tp.sig_burst_hist.keys(), tp.sig_burst_hist.values(),
                 xerr=0.5, yerr=np.sqrt(np.array(tp.sig_burst_hist.values())), label="Qi's signal",
                 fmt='^', color='c', ecolor='c', capthick=0)
    plt.errorbar(tp.avg_bkg_hist.keys(), tp.avg_bkg_hist.values(),
                 xerr=0.5, yerr=np.sqrt(np.array(tp.avg_bkg_hist.values())), label="Qi's background",
                 fmt='>', color='m', ecolor='m', capthick=0)
    plt.title("run "+str(runid)+" "+str(window_size)+"-s window")
    plt.xlabel("Burst size")
    plt.ylabel("Counts")
    if ylog:
        plt.yscale('log')
    plt.legend()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    if show:
        plt.show()
    else:
        plt.clf()


def compare_run_ratio(runid, window_size=10, outfile=None, ylog=True, pkldir="batch_all_scramble_all_events", show=True):
    if window_size==1:
        test_pbh = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window1.0-s.pkl")
        tp = test_pbh.pbhs[0]
    elif window_size==2:
        test_pbh2 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window2.0-s.pkl")
        tp = test_pbh2.pbhs[0]
    elif window_size==5:
        test_pbh5 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window5.0-s.pkl")
        tp = test_pbh5.pbhs[0]
    elif window_size==10:
        test_pbh10 = load_pickle(pkldir+"/pbhs_bkg_method_scramble_run"+str(runid)+"_window10.0-s.pkl")
        tp = test_pbh10.pbhs[0]



    fname="/raid/reedbuck/archs/PBHAnalysis_cori/MoreFinalResults/run_"+str(runid)+"_burst.root"

    rf=ROOT.TFile(fname, "read")
    if window_size==1:
        h10b = rf.Get("1.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("1.0-second_run_" + str(runid) + "_Data");
    elif window_size == 2:
        h10b = rf.Get("2.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("2.0-second_run_" + str(runid) + "_Data");
    elif window_size == 5:
        h10b = rf.Get("5.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("5.0-second_run_" + str(runid) + "_Data");
    elif window_size == 10:
        h10b=rf.Get("10.0-second_run_"+str(runid)+"_Background");
        h10=rf.Get("10.0-second_run_"+str(runid)+"_Data");

    x=np.zeros(6)
    y=np.zeros(6)
    dy=np.zeros(6)

    xB=np.zeros(6)
    yB=np.zeros(6)
    dyB=np.zeros(6)

    for i in range(6):
        xB[i]=h10b.GetBinCenter(i+1)
        yB[i]=h10b.GetBinContent(i+1)
        dyB[i]=h10b.GetBinError(i+1)
        x[i] = h10.GetBinCenter(i + 1)
        y[i] = h10.GetBinContent(i + 1)
        dy[i] = h10.GetBinError(i + 1)


    plt.errorbar(x, y*1.0/tp.sig_burst_hist.values(), xerr=0.5, yerr=np.sqrt(dy**2+np.array(tp.sig_burst_hist.values())), label="Ratio of signals",
                 fmt='v', color='r', ecolor='r', capthick=0)
    plt.errorbar(xB, yB*1.0/tp.avg_bkg_hist.values(), xerr=0.5, yerr=np.sqrt(dyB**2+np.array(tp.avg_bkg_hist.values())), label="Ratio of background",
                 fmt='<', color='b', ecolor='b', capthick=0)
    plt.title("run "+str(runid)+" "+str(window_size)+"-s window")
    plt.xlabel("Burst size")
    plt.ylabel("Simon's Counts / Qi's Counts")
    if ylog:
        plt.yscale('log')
    plt.legend()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    if show:
        plt.show()
    else:
        plt.clf()


def combine_pbhs_from_pickle_list(list_of_pbhs_pickle, outfile="pbhs_combined"):
    list_df = pd.read_csv(list_of_pbhs_pickle, header=None)
    list_df.columns = ["pickle_file"]
    list_of_pbhs = list_df.pickle_file.values
    first_pbh = load_pickle(list_of_pbhs[0])
    window_size = first_pbh.window_size
    pbhs_combined = Pbh_combined(window_size)
    pbhs_combined.rho_dots=first_pbh.rho_dots
    for pbhs_pkl_ in list_of_pbhs:
        pbhs_ = load_pickle(pbhs_pkl_)
        assert window_size==pbhs_.window_size, "The input list contains pbhs objects with different window sizes!"
        pbhs_combined.add_pbh(pbhs_)
    dump_pickle(pbhs_combined, outfile+"_window"+str(window_size)+"-s_"+str(list_of_pbhs.shape[0])+"runs.pkl")
    return pbhs_combined

def qsub_job_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                     script_dir = '/raid/reedbuck/qfeng/pbh/', overwrite=True, hostname=None, walltime=48):
    if window_size<1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f'%(filename, window_size))
    #data_base_dir = '/raid/reedbuck/veritas/data/'
    #script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["runNum"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "pbh.py"
            if window_size<1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.pbs' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_"+str(bkg_method)+"_run"+str(run_num)+"_window"+str(window_size)+"-s.pkl"
            #if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                #print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***'%pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    #sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            #script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#PBS -e %s\n'%os.path.join(script_dir, 'qsub_%s.err'%scriptname))
                script.write('#PBS -o %s\n'%os.path.join(script_dir, 'qsub_%s.log'%scriptname))
                script.write('#PBS -l walltime='+str(walltime)+':00:00\n')
                script.write('#PBS -l pvmem=5gb\n')
                script.write('cd %s\n'%script_dir)
                if plot:
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
            script.close()
            #isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            isend_command = 'qsub -l nodes=%s -q batch -V %s'%(hostname, scriptfullname)

            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise

def qsub_cori_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                     script_dir = '/global/cscratch1/sd/qifeng/pbh/', overwrite=True, hostname=None, walltime=48):
    if window_size<1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f'%(filename, window_size))
    #data_base_dir = '/raid/reedbuck/veritas/data/'
    #script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["runNum"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "pbh.py"
            if window_size<1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.pbs' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            #scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_"+str(bkg_method)+"_run"+str(run_num)+"_window"+str(window_size)+"-s.pkl"
            #if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                #print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***'%pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    #sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            #logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            #script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#!/bin/bash -l \n\n')
                script.write('#SBATCH -p shared \n')
                script.write('#SBATCH -N 1\n')
                script.write('#SBATCH -L SCRATCH\n')
                script.write('#SBATCH -e %s\n' % os.path.join(script_dir, 'qsub_%s.err' % scriptname))
                script.write('#SBATCH -o %s\n' % os.path.join(script_dir, 'qsub_%s.log' % scriptname))
                script.write('#SBATCH -t %s\n' % (str(walltime) + ':00:00\n'))
                # script.write('#SBATCH -l pvmem=5gb\n')
                script.write("source /global/homes/q/qifeng/.bashrc.ext")
                script.write('cd %s\n' % script_dir)
                if plot:
                    #script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    #script.write('python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    #script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
            script.close()
            #isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            #isend_command = 'qsub -l nodes=%s -q batch -V %s'%(hostname, scriptfullname)
            isend_command = 'sbatch -C haswell %s'%(scriptfullname)



            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise


def qsub_tehanu_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                     script_dir = '/a/data/tehanu/qifeng/pbh', overwrite=True, hostname=None, walltime=48):
    if window_size<1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f'%(filename, window_size))
    #data_base_dir = '/raid/reedbuck/veritas/data/'
    #script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["runNum"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "pbh.py"
            if window_size<1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.sh' % (run_num, window_size)
                condor_scriptname = 'pbhs_run%d_window_size%.4f-s.condor' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.sh'%(run_num, window_size)
                condor_scriptname = 'pbhs_run%d_window_size%d-s.condor' % (run_num, window_size)
            #scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            condor_scriptname = os.path.join(script_dir, condor_scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_"+str(bkg_method)+"_run"+str(run_num)+"_window"+str(window_size)+"-s.pkl"
            #if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                #print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***'%pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    #sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            #logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            #script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#!/bin/bash  \n\n')
                script.write('date \n')
                script.write('hostname \n')
                script.write('cd {} \n'.format(script_dir))
                script.write('pwd \n')
                script.write('shopt -s expand_aliases \n')
                script.write('source /usr/nevis/adm/nevis-init.sh \n')
                script.write('source /a/home/tehanu/qifeng/.bashrc \n')

                if plot:
                    #script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    #script.write('python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    #script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size<1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n' % (
                        pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))

                script.write('pwd \n')
                script.write('whoami \n')
                script.write('date \n')

            with open(condor_scriptname, 'w') as condor_script:
                condor_script.write('Universe  = vanilla \n')
                condor_script.write('Executable = {} \n'.format(scriptfullname))
                condor_script.write('Log = {}/condor_{}.log \n'.format('/'.join(scriptfullname.split('/')[:-2]), '.'.join(logfilename.split('.')[:-1]) ))
                condor_script.write('Error =  {}/condor_{}.err \n'.format('/'.join(scriptfullname.split('/')[:-2]), '.'.join(logfilename.split('.')[:-1]) ))
                condor_script.write('Output =  {}/condor_{}.out \n'.format('/'.join(scriptfullname.split('/')[:-2]), '.'.join(logfilename.split('.')[:-1]) ))
                condor_script.write('should_transfer_files = YES \n')
                condor_script.write('WhenToTransferOutput = ON_EXIT \n')
                #condor_script.write('Requirements = (machine == \"tehanu.nevis.columbia.edu\" || machine== \"ged.nevis.columbia.edu\" || machine== \"serret.nevis.columbia.edu\") \n')
                condor_script.write('Requirements = (machine== \"ged.nevis.columbia.edu\" || machine== \"serret.nevis.columbia.edu\") \n')
                condor_script.write('Notification = NEVER \n')
                condor_script.write('getenv = True \n')
                condor_script.write('Queue \n')

            #isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            #isend_command = 'qsub -l nodes=%s -q batch -V %s'%(hostname, scriptfullname)
            #isend_command = 'sbatch -C haswell %s'%(scriptfullname)
            isend_command = 'condor_submit {}'.format(condor_scriptname)



            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise


def filter_good_runlist(infile="batch_all_v3/runlist_Final.txt", outfile="goodruns.txt"):
    df_run = pd.read_csv(infile)
    df_run.columns = ['run']
    bad_runs = []
    no_ea = []
    for run in df_run.values.flatten():
        p_ = Pbh()
        p_.readEDfile(run)
        all_gamma_treeName = "run_" + str(run) + "/stereo/TreeWithAllGamma"
        all_gamma_tree = p_.Rfile.Get(all_gamma_treeName)
        es_ = []

        ea_Name = "run_" + str(run) + "/stereo/EffectiveAreas/gMeanEffectiveArea"
        ea = p_.Rfile.Get(ea_Name);
        try:
            ea.GetN()
        except:
            print("Empty EAs for run {}".format(run))
            no_ea.append(run)
            continue

        for i, event in enumerate(all_gamma_tree):
            if i > 100:
                break
            es_.append(event.Energy)

        if np.mean(np.asarray(es_)) == -99:
            print("Energy not filled for run {}".format(run))
            bad_runs.append(run)


    good_run = df_run[~df_run.run.isin(bad_runs)]
    good_run = good_run[~good_run.run.isin(no_ea)]
    print("Saving {} good runs to file {}".format(good_run.shape[0], outfile))
    good_run.to_csv(outfile, header=None, index=False)
    return good_run


def jackknife_runlist(runlist="pbh_1-s_scramble_all_90eff_list.txt", num_samples=5, run=True, window_size=1,
                      filetag="scramble_all_90eff", upper_burst_size=None, start_subsample=0):
    rlist=[]
    nline=0
    with open(runlist, 'r') as f:
        for line in f:
            rlist.append(line)
            nline+=1

    #chunk_length = nline//num_samples*(num_samples-1)
    for i in range(start_subsample, num_samples):
        outfilename = 'file'+str(i)+'out_of'+str(num_samples)+runlist
        outfile = open(outfilename, 'w')
        subsample = [x for k, x in enumerate(rlist) if k % num_samples != i]
        #outfile.write("".join(rlist[chunk_length*i:chunk_length*(i+1)]))
        outfile.write("".join(subsample))
        outfile.close()
        chunk_length = len(subsample)

        if run:
            pbhs_combined_cv = combine_from_pickle_list(outfilename, window_size,
                                                            filetag=filetag+'_'+str(i)+'out_of'+str(num_samples), upper_burst_size=upper_burst_size)
            shutil.copyfile('pbhs_combined_window'+str("{:.1f}".format(window_size))+'-s_'+str(chunk_length)+'runs.pkl',
                            'pbhs_combined_window'+str("{:.1f}".format(window_size))+'-s_'+str(chunk_length)+'runs_sub'+str(i)+'.pkl')
            print("Final UL is {0}".format(pbhs_combined_cv.rho_dot_ULs))


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-l","--list",dest="runlist", default=None)
    parser.add_option("-r","--run",dest="run", type="int", default=None)
    parser.add_option("-w","--window",dest="window", type="float", default=10)
    #parser.add_option("-p","--plot",dest="plot",default=False)
    parser.add_option("-p","--plot", action="store_true", dest="plot", default=False)
    parser.add_option("-b","--bkg_method", dest="bkg_method", default="scramble")
    parser.add_option("-m","--makeup", action="store_false", dest="overwrite", default=True)
    parser.add_option("-t","--walltime",dest="walltime", type="int", default=48)
    parser.add_option("-c","--cori", action="store_true", dest="cori", default=False)
    parser.add_option("--tehanu", action="store_true", dest="tehanu", default=False)
    #parser.add_option("--rho_dots",dest="rho_dots", default=np.arange(0, 2e7, 1e4))
    #parser.add_option("-inner","--innerHi",dest="innerHi",default=True)
    (options, args) = parser.parse_args()

    if options.runlist is not None:
        #print('Submitting jobs for runlist %s with search window size %.1f'%(options.runlist, options.window))
        if options.cori:
            qsub_cori_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                         bkg_method=options.bkg_method, script_dir=os.getcwd(), overwrite=options.overwrite, walltime=options.walltime)
        elif options.tehanu:
            qsub_tehanu_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                         bkg_method=options.bkg_method, overwrite=options.overwrite, walltime=options.walltime)
        else:
            qsub_job_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                         bkg_method=options.bkg_method, script_dir=os.getcwd(), overwrite=options.overwrite, walltime=options.walltime)

    if options.run is not None:
        print('\n\n#########################################')
        if options.window<1:
            print('Processing run %d with search window size %.1g' % (options.run, options.window))
        else:
            print('Processing run %d with search window size %.1f'%(options.run, options.window))
        process_one_run(options.run, options.window, bkg_method=options.bkg_method, plot=options.plot)

    #test_singlet_remover()
    #pbh = test_burst_finding(window_size=5, runNum=55480, nlines=None, N_scramble=5,
    #                         save_hist="test_burst_finding_histo", bkg_method="rando")
    #pbh = test_ll(window_size=3, runNum=55480, N_scramble=3, verbose=False, rho_dots=np.arange(1e3, 5.e5, 1000.),
    #              save_hist="test_ll", bkg_method="scramble", rando_method="avg",
    #              burst_size=2)
    #pbh = test_psf_func(Nburst=10, filename=None)

    #pbh = test_psf_func_sim(psf_width=0.05, Nsim=10000, prob="psf", Nbins=40, xlim=(0,0.5),
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_hypsec_sig.pdf")

    #pbh = test_psf_func_sim(psf_width=0.05, prob="uniform", Nsim=10000, Nbins=40, xlim=None,
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_uniform_bkg.pdf")
