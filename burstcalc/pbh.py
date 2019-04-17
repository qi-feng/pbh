import sys
import os
import random
from math import factorial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.optimize import minimize
from scipy import integrate
from scipy.special import gamma

from main.gamma_burst import Burst
from main.io import *

sys.setrecursionlimit(50000)

def combine_pbhs_from_pickle_list(list_of_pbhs_pickle, outfile="pbhs_combined"):
    '''
    Read a list of pickle file names in a csv file and combine data if from same window size.
    Also, save combined data to an output pickle file
    :param list_of_pbhs_pickle: csv file with list of pickle file names
    :param outfile: output file name
    :return: combined picke file data
    '''
    list_df = pd.read_csv(list_of_pbhs_pickle, header=None)
    list_df.columns = ["pickle_file"]
    list_of_pbhs = list_df.pickle_file.values
    first_pbh = load_pickle(list_of_pbhs[0])
    window_size = first_pbh.window_size
    pbhs_combined = PbhCombined(window_size)
    pbhs_combined.rho_dots = first_pbh.rho_dots
    for pbhs_pkl_ in list_of_pbhs:
        pbhs_ = load_pickle(pbhs_pkl_)
        assert window_size == pbhs_.window_size, "The input list contains pbhs objects with different window sizes!"
        pbhs_combined.add_pbh(pbhs_)
    dump_pickle(pbhs_combined, outfile + "_window" + str(window_size) + "-s_" + str(list_of_pbhs.shape[0]) + "runs.pkl")
    return pbhs_combined



def combine_from_pickle_list(listname, window_size, filetag="", burst_size_threshold=2, rho_dots=None,
                             upper_burst_size=None):
    pbhs_combined_all_ = combine_pbhs_from_pickle_list(listname)
    print("Total exposure time is %.2f hrs" % (pbhs_combined_all_.total_time_year * 365.25 * 24))
    pbhs_combined_all_.get_upper_limits(burst_size_threshold=burst_size_threshold, rho_dots=rho_dots,
                                        upper_burst_size=upper_burst_size)
    print("The effective volume above burst size 2 is %.6f pc^3" % (pbhs_combined_all_.effective_volumes[2]))
    print("There are %d runs in total" % (len(pbhs_combined_all_.run_numbers)))
    total_N_runs = len(pbhs_combined_all_.run_numbers)
    pbhs_combined_all_.plot_burst_hist(
        filename="burst_hists_test_" + str(filetag) + "_window" + str(window_size) + "-s_all" + str(
            total_N_runs) + "runs.png",
        title="Burst histogram " + str(window_size) + "-s window " + str(total_N_runs) + " runs", plt_log=True,
        error="Poisson")
    pbhs_combined_all_.plot_ll_vs_rho_dots(
        save_hist="ll_vs_rho_dots_test_" + str(filetag) + "_window" + str(window_size) + "-s_all" + str(
            total_N_runs) + "runs")
    return pbhs_combined_all_



class PbhCombined(Burst):
    def __init__(self, window_size):
        super(PbhCombined, self).__init__()
        # the cut on -2lnL for rho_dot that gives an observed burst size
        # !!!note how this is different from the ll_cut on the centroid in the super class
        self.delta_ll_cut = 6.63
        self.photon_df = None
        self.pbhs = []
        self.run_numbers = []
        self.sig_burst_hist = {}
        self.avg_bkg_hist = {}
        self.residual_dict = {}
        # this is important, keep track of the number of runs to average over
        self.n_runs = 0
        # total exposure in unit of year
        self.total_time_year = 0
        # self.effective_volume = 0.0
        self.effective_volumes = {}
        self.minimum_lls = {}
        self.rho_dot_ULs = {}
        # Make the class for a specific window size
        self.window_size = window_size
        # Some global parameters
        self.bkg_method = "scramble_times"
        self.rando_method = "avg"
        self.N_scramble = 10
        self.verbose = False
        self.burst_sizes_set = set()
        self.rho_dots = np.arange(0., 3.e5, 100)

    def change_window_size(self, window_size):
        self.window_size = window_size
        # a bunch of stuff needs re-initialization
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
        # analyze again:
        if isinstance(orig_pbhs[0], PbhCombined):
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
                    _sig_burst_hist, _sig_burst_dict = pbh_.signal_burst_search(window_size=self.window_size,
                                                                                verbose=self.verbose)
                    pbh_.estimate_bkg_burst(window_size=self.window_size,
                                                                              rando_method=self.rando_method,
                                                                              method=self.bkg_method, copy=True,
                                                                              n_scramble=self.N_scramble,
                                                                              return_burst_dict=True,
                                                                              verbose=self.verbose)
                    pbhs_.do_step2345(pbh_)
                self.do_step2345(pbhs_)
        else:
            for pbh_ in orig_pbhs:
                _sig_burst_hist, _sig_burst_dict = pbh_.signal_burst_search(window_size=self.window_size,
                                                                            verbose=self.verbose)
                pbh_.estimate_bkg_burst(window_size=self.window_size,
                                                                          rando_method=self.rando_method,
                                                                          method=self.bkg_method, copy=True,
                                                                          n_scramble=self.N_scramble,
                                                                          return_burst_dict=True,
                                                                          verbose=self.verbose)
                self.do_step2345(pbh_)

        rho_dot_ULs = self.get_upper_limits()
        return rho_dot_ULs

    def add_pbh(self, pbh):
        # When adding a new run, we want to update:
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
        self.run_number = pbh.run_number
        self.run_numbers.append(pbh.run_number)

    def do_step2345(self, pbh):
        # 2.
        if not hasattr(pbh, 'total_time_year'):
            pbh.total_time_year = (pbh.tOn * (1. - pbh.DeadTimeFracOn)) / 31536000.
        previous_total_time_year = self.total_time_year
        self.total_time_year += pbh.total_time_year
        # 3 and 4 and residual
        current_all_burst_sizes = self.get_all_burst_sizes()
        new_all_burst_sizes = current_all_burst_sizes.union(
            set(k for dic in [pbh.sig_burst_hist, pbh.avg_bkg_hist] for k in dic.keys()))
        self.burst_sizes_set = new_all_burst_sizes
        for key_ in new_all_burst_sizes:
            # first zero-pad the new pbh hists with all possible burst sizes
            key_ = int(key_)
            if key_ not in pbh.sig_burst_hist:
                pbh.sig_burst_hist[key_] = 0
            if key_ not in pbh.avg_bkg_hist:
                pbh.avg_bkg_hist[key_] = 0
        for key_ in new_all_burst_sizes:
            # now add all the bursts in pbh to self
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
                # self.effective_volumes[key_] = pbh.total_time_year * pbh.V_eff(key_, self.window_size)
                # self.effective_volumes[key_] = self.V_eff(key_, self.window_size)
                self.effective_volumes[key_] = 0.0
                for pbh_ in self.pbhs:
                    # This already includes the new pbh object
                    Veff_ = pbh_.V_eff(key_, self.window_size, verbose=False)
                    self.effective_volumes[key_] += pbh_.total_time_year * Veff_ * 1.0
                new_total_time = 1.0 * (previous_total_time_year + pbh.total_time_year)
                self.effective_volumes[key_] = self.effective_volumes[key_] / new_total_time
            else:
                # self.effective_volumes[key_] already exists, so don't need to calculate again, this is faster
                new_total_time = 1.0 * (previous_total_time_year + pbh.total_time_year)
                self.effective_volumes[key_] = (self.effective_volumes[
                                                    key_] * previous_total_time_year + pbh.total_time_year * pbh.V_eff(
                    key_, self.window_size)) / new_total_time
            # if key_ not in pbh.effective_volumes:
            #    pbh.effective_volumes[key_] = pbh.V_eff(key_, self.window_size)

        assert self.burst_sizes_set == self.get_all_burst_sizes(), "Something is wrong when adding a pbh to pbh combined"

    # Override V_eff for the combiner class
    def V_eff(self, burst_size, t_window, verbose=False):
        assert t_window == self.window_size, "You are asking for an effective volume for a different window size."
        if burst_size not in self.effective_volumes:
            # recalculate effective volume from each single pbh class in the combiner
            self.effective_volumes[burst_size] = 0.0
            total_time = 0.0
            for pbh_ in self.pbhs:
                Veff_ = pbh_.V_eff(burst_size, self.window_size, verbose=False)
                self.effective_volumes[burst_size] += pbh_.total_time_year * Veff_ * 1.0
                total_time += pbh_.total_time_year
            self.effective_volumes[burst_size] = self.effective_volumes[burst_size] / total_time * 1.0
        return self.effective_volumes[burst_size]

    def get_all_burst_sizes(self):
        # returns a set, not a dict
        all_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        return all_burst_sizes

    def add_run(self, run_number):
        pbh_ = Pbh()
        pbh_.get_tree_with_all_gamma(run_number=run_number)
        _sig_burst_hist, _sig_burst_dict = pbh_.signal_burst_search(window_size=self.window_size, verbose=self.verbose)
        pbh_.estimate_bkg_burst(window_size=self.window_size,
                                                                  rando_method=self.rando_method,
                                                                  method=self.bkg_method, copy=True,
                                                                  n_scramble=self.N_scramble)
        pbh_.get_run_summary()
        self.add_pbh(pbh_)
        # self.run_numbers.append(run_number)

    def n_excess(self, rho_dot, Veff, verbose=False):
        # eq 8.8, or maybe more appropriately call it n_expected
        if not hasattr(self, 'total_time_year'):
            self.total_time_year = (self.tOn * (1. - self.DeadTimeFracOn)) / 31536000.
        # n_ex = 1.0 * rho_dot * self.total_time_year * Veff
        # because the burst likelihood cut -9.5 is at 90% CL
        n_ex = 0.9 * rho_dot * self.total_time_year * Veff
        if verbose:
            print("The value of the expected number of bursts (eq 8.9) is %.2f" % n_ex)
        return n_ex

    # Override get_ll so that it knows where to find the effective volume
    def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=None):
        # def get_ll(self, rho_dot, burst_size_threshold, t_window, verbose=False, upper_burst_size=100):
        # eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
        if upper_burst_size is None:
            all_burst_sizes = self.get_all_burst_sizes()
        else:
            all_burst_sizes = range(burst_size_threshold, upper_burst_size + 1)
        ll_ = 0.0
        sum_nb = 0
        self.good_burst_sizes = []  # use this to keep burst sizes that are large enough so that $n_b > \sum_b n_{b+1}$
        for burst_size in np.sort(np.array(list(all_burst_sizes)))[::-1]:
            # for burst_size in all_burst_sizes:
            if burst_size >= burst_size_threshold:
                # Veff_ = self.V_eff(burst_size, t_window, verbose=verbose)
                # print("Burst size %d " % burst_size)
                # Veff_ = self.effective_volumes[burst_size]
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
                    # print("###############################################################################")
                    print(
                        '''Adding -2lnL at burst size %d, for search window %.1f and 
                        rate density %.1f, so far -2lnL = %.2f''' % (
                            burst_size, t_window, rho_dot, ll_))
                    # print("###############################################################################")
        if verbose:
            print("###############################################################################")
            print("-2lnL above burst size %d, for search window %.1f and rate density %.1f is %.2f" % (
                burst_size_threshold, t_window, rho_dot, ll_))
            print("###############################################################################")
        return ll_

    # @autojit
    def get_minimum_ll(self, burst_size, t_window, rho_dots=np.arange(0., 3.e5, 100), return_arrays=True,
                       # verbose=False, upper_burst_size=100):
                       verbose=False, upper_burst_size=None):
        # search rho_dots for the minimum -2lnL for burst size >= burst_size
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        min_ll_ = 1.e5
        rho_dot_min_ll_ = -1.0
        if return_arrays:
            lls_ = np.zeros(rho_dots.shape[0]).astype('float64')
        i = 0
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

    def get_upper_limits(self, burst_size_threshold=2, rho_dots=None, upper_burst_size=None):
        print("Getting UL for burst size above %d..." % burst_size_threshold)
        if rho_dots is None:
            rho_dots = self.rho_dots
        minimum_rho_dot, minimum_ll, rho_dots, lls = self.get_minimum_ll(burst_size_threshold, self.window_size,
                                                                         rho_dots=rho_dots, verbose=self.verbose,
                                                                         upper_burst_size=upper_burst_size)
        self.minimum_lls[burst_size_threshold] = minimum_ll
        self.rho_dot_ULs[burst_size_threshold], ll_UL_ = self.get_ul_rho_dot(rho_dots, lls, minimum_ll, margin=1.e-5)
        return self.rho_dot_ULs

    def plot_ll_vs_rho_dots(self, save_hist="ll_vs_rho_dots", xlog=True, grid=True, plot_hline=True, show=False,
                            ylim=(0, 25)):
        rho_dots = self.rho_dots
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for b_ in self.burst_sizes_set:
            if b_ == 1:
                continue
            minimum_rho_dot, minimum_ll, rho_dots, lls = self.get_minimum_ll(b_, self.window_size, rho_dots=rho_dots,
                                                                             verbose=self.verbose)
            plt.plot(rho_dots, lls - minimum_ll,
                     label="burst size " + str(b_) + ", " + str(self.window_size) + "-s window")
        # plt.axvline(x=minimum_rho_dot, color="b", ls="--",
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
        filename = save_hist + "_" + str(self.n_runs) + "runs.png"
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        print("Done!")

    def process_run_list(self, filename="pbh_runlist.txt"):
        runlist = pd.read_csv(filename, header=None)
        runlist.columns = ["run_number"]
        self.runlist = runlist.runNum.values
        self.bad_runs = []
        for run_ in self.runlist:
            try:
                self.add_run(run_)
                print("Run %d processed." % run_)
            except:
                print("*** Bad run: %d ***" % run_)
                # raise
                self.bad_runs.append(run_)
                self.runlist = self.runlist[np.where(self.runlist != run_)]
        return self.get_upper_limits()

    def save(self, filename):
        # save_hdf5(self, filename)
        dump_pickle(self, filename)
