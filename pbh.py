__author__ = 'qfeng'
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy import stats
import random
import cPickle as pickle
from scipy.special import gamma
from math import factorial
import tables

import sys

sys.setrecursionlimit(50000)

try:
    import ROOT
    ROOT.PyConfig.StartGuiThread = False
except:
    print "Can't import ROOT, no related functionality possible"

import time

try:
    from numba import jit, autojit
except:
    print("Numba not installed")

def deg2rad(deg):
    return deg / 180. * np.pi


def rad2deg(rad):
    return rad * 180. / np.pi

class Pbh(object):
    def __init__(self):
        # the cut on -2lnL, consider smaller values accepted for events coming from the same centroid
        self.ll_cut = -9.5
        # set the hard coded PSF width table from the hyperbolic secant function
        # 4 rows are Energy bins 0.08 to 0.32 TeV (row 0), 0.32 to 0.5 TeV, 0.5 to 1 TeV, and 1 to 50 TeV
        # 3 columns are Elevation bins 50-70 (column 0), 70-80 80-90 degs
        self.psf_lookup = np.zeros((4, 3))
        self.E_grid = np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.EL_grid = np.array([50.0, 70.0, 80., 90.])
        # for later reference
        # self.E_bins=np.digitize(self.BDT_ErecS, self.E_grid)-1
        #self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
        #  0.08 to 0.32 TeV
        self.psf_lookup[0, :] = np.array([0.052, 0.051, 0.05])
        #  0.32 to 0.5 TeV
        self.psf_lookup[1, :] = np.array([0.047, 0.042, 0.042])
        #   0.5 to 1 TeV
        self.psf_lookup[2, :] = np.array([0.041, 0.035, 0.034])
        #   1 to 50 TeV
        self.psf_lookup[3, :] = np.array([0.031, 0.028, 0.027])
        self._burst_dict = {}  #{"Burst #": [event # in this burst]}, for internal use
        self.VERITAS_deadtime = 0.33e-3  # 0.33ms

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

    def readEDfile(self, runNum=None, filename=None):
        self.runNum = runNum
        self.filename = str(runNum) + ".anasum.root"
        if not os.path.isfile(self.filename) and filename is not None:
            if os.path.isfile(filename):
                self.filename = filename
        self.Rfile = ROOT.TFile(self.filename, "read");

    def get_TreeWithAllGamma(self, runNum=None, E_lo_cut=0.08, E_hi_cut=50.0, EL_lo_cut=50.0, nlines=None):
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
        i_gamma = 0
        for i, event in enumerate(all_gamma_tree):
            #if nlines is not None:
            #    if breaker >= nlines:
            #        break
            #    breaker += 1
            #time_index=np.argmax(ptTime>event.Time)
            #making cut:
            #this is quite essential to double check!!!
            if event.Energy < E_lo_cut or event.Energy > E_hi_cut or event.TelElevation < EL_lo_cut or event.IsGamma==0:
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

        print("There are %d events, %d of which are gamma-like" % (N_,i_gamma))
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

    def scramble(self, copy=False):
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        ts_ = self.photon_df.ts.values
        random.shuffle(ts_)
        self.photon_df.at[:, 'ts'] = ts_
        # re-init _burst_dict for counting
        self._burst_dict = {}
        # print self.photon_df.head()
        #print self.photon_df.ts.shape, self.photon_df.ts
        #sort!
        self.photon_df=self.photon_df.sort('ts')
        return ts_

    def t_rando(self, copy=False, rate="avg"):
        """
        throw Poisson distr. ts based on the original ts,
        use 1/delta_t as the expected Poisson rate for each event
        """
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        if copy:
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble
            if not hasattr(self, 'photon_df_orig'):
                self.photon_df_orig = self.photon_df.copy()
        if rate=="cell":
            delta_ts = np.diff(self.photon_df.ts)
        #for i, _delta_t in enumerate(delta_ts):
        N = self.photon_df.shape[0]
        rate_expected= N*1.0/(self.photon_df.ts.values[-1]-self.photon_df.ts.values[0])
        print("Mean expected rate is %.2f" % rate_expected)
        for i in range(N-1):
            if rate=="cell":
                rate_expected = 1. / delta_ts[i]
            #elif rate=="avg":

            # draw a rando!
            _rando_delta_t = np.random.exponential(rate_expected)
            inf_loop_preventer = 0
            inf_loop_bound = 100
            while _rando_delta_t < self.VERITAS_deadtime:
                _rando_delta_t = np.random.exponential(rate_expected)
                inf_loop_preventer += 1
                if inf_loop_preventer > inf_loop_bound:
                    print "Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,"
                    print "you'd better check your time unit or something..."
            self.photon_df.at[i + 1, 'ts'] = self.photon_df.ts[i] + _rando_delta_t
        #naturally sorted
        # re-init _burst_dict for counting
        self._burst_dict = {}
        return self.photon_df.ts


    @autojit
    def psf_func(self, theta2, psf_width, N=100):
        return 1.71 * N / 2. / np.pi / psf_width ** 2 / np.cosh(np.sqrt(theta2) / psf_width)
        # equivalently:
        #return (stats.hypsecant.pdf(np.sqrt(theta2s)/psf_width)*1.71/2./psf_width**2)

    def psf_cdf(self, psf_width, fov=1.75):
        """
        :param psf_width: same as psf_func
        :param fov: given so that we calculate cdf from 0 to fov
        :return:
        """
        theta2s = np.arange(0, fov, 0.01)
        cdf = np.cumsum(self.psf_func(theta2s, psf_width, N=1))
        cdf = cdf / np.max(cdf)
        return cdf

    # use hard coded width table from the hyperbolic secant function
    def get_psf(self, E=0.1, EL=80):
        E_bin = np.digitize(E, self.E_grid) - 1
        EL_bin = np.digitize(EL, self.EL_grid) - 1
        return self.psf_lookup[E_bin, EL_bin]

    @autojit
    def get_psf_lists(self):
        """
        This thing is slow...
        :return: nothing but filles photon_df.psfs, a number that is repeatedly used later
        """
        if not hasattr(self, 'photon_df'):
            print "Call get_TreeWithAllGamma first..."
        ###QF:
        print "getting psf"
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

    @autojit
    def get_angular_distance(self, coord1, coord2):
        """
        coord1 and coord2 are in [ra, dec] format in degrees
        """
        return rad2deg(np.arccos(np.sin(deg2rad(coord1[1])) * np.sin(deg2rad(coord2[1]))
                                 + np.cos(deg2rad(coord1[1])) * np.cos(deg2rad(coord2[1])) *
                                 np.cos(deg2rad(coord1[0]) - deg2rad(coord2[0]))))

    @autojit
    def get_all_angular_distance(self, coords, cent_coord):
        assert coords.shape[1] == 2
        dists = np.zeros(coords.shape[0])
        for i, coord in enumerate(coords):
            dists[i] = self.get_angular_distance(coord, cent_coord)
        return dists

    def gen_one_random_coords(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        _phi = np.random.random() * np.pi * 2.
        _ra = cent_coord[0] + np.sin(_phi) * theta
        _dec = cent_coord[1] + np.cos(_phi) * theta
        return np.array([_ra, _dec])

    def gen_one_random_theta(self, psf_width, prob="psf", fov=1.75):
        """
        :prob can be either "psf" that uses the hyper-sec function, or "uniform", or "gauss"
        """
        if prob.lower() == "psf" or prob == "hypersec" or prob == "hyper secant":
            #_rand_theta = np.random.random()*fov
            _rand_test_cdf = np.random.random()
            _thetas = np.arange(0, fov, 0.01)
            _theta2s = _thetas ** 2
            #_theta2s=np.arange(0, fov*fov, 0.0001732)
            _psf_pdf = self.psf_func(_theta2s, psf_width, N=1)
            _cdf = np.cumsum(_psf_pdf - np.min(_psf_pdf))
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

    @autojit
    def centroid_log_likelihood(self, cent_coord, coords, psfs):
        """
        returns ll=-2*ln(L)
        """
        ll = 0
        dists = self.get_all_angular_distance(coords, cent_coord)
        theta2s = dists ** 2
        ll = -2. * np.sum(np.log(psfs)) + np.sum(np.log(1. / np.cosh(np.sqrt(theta2s) / psfs)))
        ll += psfs.shape[0] * np.log(1.71 / np.pi)
        ll = -2. * ll
        #return ll
        #Normalized by the number of events!
        return ll / psfs.shape[0]

    @autojit
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
        if ll_centroid <= self.ll_cut:
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
            print "You started a burst search while there are already things in _burst_dict, now make it empty"
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
            slice_index, singlet_slice_index = self.singlet_remover(np.array(slice_index[0]))

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
                #If there is a burst of a subset of events, it's been taken care of, now take care of the outlier slice
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

    @autojit
    def singlet_remover(self, slice_index):
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
        coord_slice = np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
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
            np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
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
                    np.concatenate([self.photon_df.RAs.reshape(N_, 1), self.photon_df.Decs.reshape(N_, 1)], axis=1)[
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

    @autojit
    def burst_counting(self):
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
            self.burst_counting()

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

    def estimate_bkg_burst(self, window_size=1, method="scramble", copy=True, n_scramble=1, rando_method="avg",
                           return_burst_dict=False, verbose=False):
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
                self.scramble(copy=copy)
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
        #eq 8.3 with no acceptance (I in eq 8.7); EA normalized to the unit of pc^2
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
        energy_cut_indices = np.where((number_expected[:, 0]>=80.) & (number_expected[:, 0]<=50000.))
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
        n_ex = 1.0 * rho_dot * self.total_time_year * Veff
        if verbose:
            print("The value of the expected number of bursts (eq 8.9) is %.2f" % n_ex)
        return n_ex

    @autojit
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

    def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False):
        #eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
        all_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        ll_ = 0.0
        for burst_size in all_burst_sizes:
            if burst_size >= burst_size_threshold:
                Veff_ = self.V_eff(burst_size, t_window, verbose=verbose)
                n_expected_ = self.n_excess(rho_dot, Veff_, verbose=verbose)
                if burst_size not in self.sig_burst_hist:
                    self.sig_burst_hist[burst_size] = 0
                if burst_size not in self.avg_bkg_hist:
                    self.avg_bkg_hist[burst_size] = 0
                n_on_ = self.sig_burst_hist[burst_size]
                n_off_ = self.avg_bkg_hist[burst_size]
                ll_ += self.ll(n_on_, n_off_, n_expected_)
                if verbose:
                    #print("###############################################################################")
                    print("Adding -2lnL at burst size %d, for search window %.1f and rate density %.1f, so far -2lnL = %.2f" % (burst_size, t_window, rho_dot, ll_))
                    #print("###############################################################################")
        if verbose:
            print("###############################################################################")
            print("-2lnL above burst size %d, for search window %.1f and rate density %.1f is %.2f" % (burst_size_threshold, t_window, rho_dot, ll_))
            print("###############################################################################")
        return ll_

    def get_ll_vs_rho_dot(self, burst_size, t_window, rho_dots=np.arange(1e5, 2.e6, 100), verbose=False):
        #plot a vertical slice of Fig 8-4, for a given burst size and search window, scan through rho_dot and plot -2lnL
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        lls_ = np.zeros(rho_dots.shape[0])
        for i, rho_dot_ in enumerate(rho_dots):
            lls_[i] = self.get_ll(rho_dot_, burst_size, t_window, verbose=verbose)
        return rho_dots, lls_

    @autojit
    def get_minimum_ll(self, burst_size, t_window, rho_dots=np.arange(0., 3.e5, 100), return_arrays=True,
                       verbose=False):
        #search rho_dots for the minimum -2lnL
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        min_ll_ = 1.e5
        rho_dot_min_ll_ = -1.0
        if return_arrays:
            lls_ = np.zeros(rho_dots.shape[0])
        i=0
        for rho_dot_ in rho_dots:
            ll_ = self.get_ll(rho_dot_, burst_size, t_window, verbose=verbose)
            if ll_ < min_ll_:
                min_ll_ = ll_
                rho_dot_min_ll_ = rho_dot_
            if return_arrays:
                lls_[i] = ll_
                i += 1
        if return_arrays:
            return rho_dot_min_ll_, min_ll_, rho_dots, lls_
        return rho_dot_min_ll_, min_ll_

    def get_ul_rho_dot(self, rho_dots, lls_, min_ll_):
        ll_99 = 6.63
        ul_99_idx = (np.abs(lls_-min_ll_-ll_99)).argmin()
        ul_99_idx_all = np.where(abs(lls_-lls_[ul_99_idx])<1e-5)
        if ul_99_idx_all[0].shape[0]==0:
            print("Can't find 99% UL!")
            raise
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
            bkg_err = np.zeros(np.array(self.avg_bkg_hist.values()).hape[0])
        elif error=="Poisson":
            sig_err = np.sqrt(np.array(self.sig_burst_hist.values()).astype('float'))
            bkg_err = np.sqrt(np.array(self.avg_bkg_hist.values()).astype('float'))
        elif error.lower()=="std":
            sig_err = np.sqrt(np.array(self.sig_burst_hist.values()).astype('float'))
            all_bkg_burst_sizes = set(k for dic in self.bkg_burst_hists for k in dic.keys())
            bkg_err = np.zeros(sig_err.shape[0])
            for key_ in all_bkg_burst_sizes:
                key_ = float(key_)
                bkg_err[key_] = np.std(np.array([d[key_] for d in self.bkg_burst_hists if key_ in d]))

        ax1.errorbar(self.sig_burst_hist.keys(), self.sig_burst_hist.values(), xerr=0.5,
                         yerr=sig_err, fmt='bs', capthick=0,
                         label="Data")
        ax1.errorbar(self.avg_bkg_hist.keys(), self.avg_bkg_hist.values(), xerr=0.5,
                         yerr=bkg_err, fmt='rv', capthick=0,
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
        ax2.errorbar(residual_dict.keys(), residual_dict.values(), xerr=0.5, yerr=res_err, fmt='bs', capthick=0,
                     label="Residual")
        ax2.axhline(y=0, color='gray', ls='--')
        ax2.set_xlabel("Burst size")
        ax2.set_ylabel("Counts")
        #plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        #plt.yscale('log')
        plt.legend(loc='best')
        if filename is not None:
            plt.savefig(filename)
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
                    label=None):
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = plt.subplot(111)
        ax.plot(coords[:, 0], coords[:, 1], color + '.')
        label_flag = False
        for coor, E_, EL_ in zip(coords, Es, ELs):
            if label_flag == False:
                circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False, label=label)
                label_flag = True
            else:
                circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
            ax.add_patch(circ)

        label_flag = False
        if fov is not None and fov_center is not None:
            circ_fov = plt.Circle(fov_center, radius=fov, color=fov_color, fill=False)
            ax.add_patch(circ_fov)
            ax.set_xlim(fov_center[0] - fov * 1.1, fov_center[0] + fov * 1.1)
            ax.set_ylim(fov_center[1] - fov * 1.1, fov_center[1] + fov * 1.1)
        if cent_coords is not None:
            #circ_cent=plt.Circle(cent_coords, radius=cent_radius, color=cent_color, fill=False)
            #ax.add_patch(circ_cent)
            ax.plot(cent_coords[0], cent_coords[1], marker=cent_marker, ms=cent_ms, markeredgewidth=cent_mew,
                    color=color)

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
        self.effective_volume = 0.0
        #Make the class for a specific window size
        self.window_size = window_size
        #Some global parameters
        self.bkg_method = "scramble"
        self.rando_method = "cell"
        self.N_scramble = 10
        self.verbose = False

    def add_pbh(self, pbh):
        #When adding a new run, we want to update:
        # 1. self.n_runs
        # 2. self.total_time_year
        # 3. self.sig_burst_hist
        # 4. self.avg_bkg_hist
        # 5. self.effective_volume
        self.pbhs.append(pbh)
        # 1.
        previous_n_runs = self.n_runs
        self.n_runs += 1
        # 2.
        if not hasattr(pbh, 'total_time_year'):
            pbh.total_time_year = (pbh.tOn*(1.-pbh.DeadTimeFracOn))/31536000.
        previous_total_time_year = self.total_time_year
        self.total_time_year += pbh.total_time_year
        # 3 and 4 and residual
        current_all_burst_sizes = self.get_all_burst_sizes()
        new_all_burst_sizes = current_all_burst_sizes.union(set(k for dic in [pbh.sig_burst_hist, pbh.avg_bkg_hist] for k in dic.keys()))
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
        self.effective_volume = self.effective_volume * previous_total_time_year + pbh.total_time_year * pbh.V_eff()

    def get_all_burst_sizes(self):
        #returns a set, not a dict
        all_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        return all_burst_sizes

    def add_run(self, runNum):
        pbh_ = Pbh()
        pbh_.get_TreeWithAllGamma(runNum=runNum, nlines=None)
        _sig_burst_hist, _sig_burst_dict = pbh.sig_burst_search(window_size=self.window_size, verbose=self.verbose)
        _avg_bkg_hist, _bkg_burst_dicts = pbh.estimate_bkg_burst(window_size=self.window_size, rando_method=self.rando_method,
                                                               method=self.bkg_method,copy=True, n_scramble=self.N_scramble,
                                                               return_burst_dict=True, verbose=self.verbose)
        pbh_.getRunSummary()
        self.add_pbh(pbh_)
        self.runNums.append(runNum)

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


def test_sim_likelihood(Nsim=1000, N_burst=3, filename=None, sig_bins=50, bkg_bins=100, ylog=True):
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    #spec sim:
    index = -2.5
    E_min = 0.08
    E_max = 50.0
    #Burst size to visualize
    #N_burst = 10
    EL = 15
    pl_nu = powerlaw(index, E_min, E_max)

    #Nsim = 1000
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
    return pbh


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
    plt.savefig(filename, dpi=150)
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
        plt.savefig(filename, dpi=150)
    plt.show()


def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=55480, nlines=1000)
    print(pbh.photon_df.head())
    return pbh


if __name__ == "__main__":
    #test_singlet_remover()
    #pbh = test_burst_finding(window_size=5, runNum=55480, nlines=None, N_scramble=5,
    #                         save_hist="test_burst_finding_histo", bkg_method="rando")
    pbh = test_ll(window_size=3, runNum=55480, N_scramble=3, verbose=False, rho_dots=np.arange(1e3, 5.e5, 1000.),
                  save_hist="test_ll", bkg_method="scramble", rando_method="avg",
                  burst_size=2)
    #pbh = test_psf_func(Nburst=10, filename=None)

    #pbh = test_psf_func_sim(psf_width=0.05, Nsim=10000, prob="psf", Nbins=40, xlim=(0,0.5),
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_hypsec_sig.pdf")

    #pbh = test_psf_func_sim(psf_width=0.05, prob="uniform", Nsim=10000, Nbins=40, xlim=None,
    #                        filename="/Users/qfeng/Data/veritas/pbh/reedbuck_plots/test_sim_psf/sim_psf_theta2hist_uniform_bkg.pdf")