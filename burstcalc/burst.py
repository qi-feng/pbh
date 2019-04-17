import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

from scipy.optimize import minimize

from astropy import units as u
from astropy.coordinates import SkyCoord

from burstcalc.veritas import VeritasFile


class BurstFile(VeritasFile):
    def __init__(self, run_number, data_dir, using_ed=True, num_ev_to_read=None, debug=False,
                 veritas_deadtime_ms=0.33e-3):
        super().__init__(run_number=run_number, data_dir=data_dir, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                         debug=debug)
        # TODO: remove me - this should come from the run file
        # or if it is only used in the sims then it should come from config file or ...
        self.veritas_deadtime_ms = veritas_deadtime_ms

        # the cut on -2lnL, consider smaller values accepted for events coming from the same centroid
        # selected based on sims at 90% efficiency
        # TODO: document where this came from better

        # self.ll_cut = -9.5

        # TODO: document where the LL cut came from better

        # self.ll_cut_dict = {2:-9.11,3:-9.00,4:-9.01, 5:-9.06, 6:-9.12, 7:-9.16, 8:-9.19, 9:-9.21, 10:-9.25}
        # Dec=0 after cos correction
        # self.ll_cut_dict = {2:-8.81,3:-8.69,4:-8.80, 5:-8.82, 6:-8.85, 7:-8.90, 8:-8.92, 9:-8.98, 10:-8.99}
        # Dec=80 after cos correction
        # self.ll_cut_dict = {2:-8.81,3:-8.72,4:-8.76, 5:-8.83, 6:-8.86, 7:-8.88, 8:-8.95, 9:-8.96, 10:-8.99}
        # Dec=80 after cos correction, from fit
        # self.ll_cut_dict = {2:-8.83,3:-8.73,4:-8.76, 5:-8.81, 6:-8.86, 7:-8.90, 8:-8.94, 9:-8.97, 10:-8.99}
        # Dec=80 after cos correction, from fit, new cumtrapz integration
        # self.ll_cut_dict = {2:-8.637,3:-8.55,4:-8.564, 5:-8.614, 6:-8.656, 7:-8.7, 8:-8.737, 9:-8.767, 10:-8.794}
        # Dec=80 after cos correction, new cumtrapz integration, theta2 instead of theta
        # self.ll_cut_dict = {2:-6.68,3:-6.74,4:-6.71, 5:-6.76, 6:-6.8, 7:-6.84, 8:-6.88, 9:-6.92, 10:-6.96}
        # new cuts using scrambled data (2017-06-13) 6 runs mean:
        self.ll_cut_dict = {2: -6.95, 3: -6.95, 4: -6.96, 5: -6.99, 6: -7.03, 7: -7.07, 8: -7.12, 9: -7.16, 10: -7.18}

        # if not in the above dict then use the following as a default value
        self.ll_cut = -8.6

        # TODO: document where the PSF came from better
        # TODO: load PSF from a config file so that we can make it depend upon the cuts used

        # set the hard coded PSF width table from the hyperbolic secant function
        # 4 rows are Energy bins 0.08 to 0.32 TeV (row 0), 0.32 to 0.5 TeV, 0.5 to 1 TeV, and 1 to 50 TeV
        # 3 columns are Elevation bins 50-70 (column 0), 70-80 80-90 degs
        self.psf_lookup = np.zeros((4, 3)).astype('float')
        self.energy_grid_tev = np.array([0.08, 0.32, 0.5, 1.0, 50.0])
        self.elevation_grid_deg = np.array([50.0, 70.0, 80., 90.])

        # TODO: document where this came from better

        # for later reference
        # self.E_bins=np.digitize(self.BDT_ErecS, self.energy_grid_tev)-1
        # self.Z_bins=np.digitize((90.-self.BDT_Elevation), self.Zen_grid)-1
        #  0.08 to 0.32 TeV
        #  the 3 elements are Elevation 50-70, 70-80 80-90 degs
        self.psf_lookup[0, :] = np.array([0.052, 0.051, 0.05])
        #  0.32 to 0.5 TeV
        self.psf_lookup[1, :] = np.array([0.047, 0.042, 0.042])
        #   0.5 to 1 TeV
        self.psf_lookup[2, :] = np.array([0.041, 0.035, 0.034])
        #   1 to 50 TeV
        self.psf_lookup[3, :] = np.array([0.031, 0.028, 0.027])

        # TODO: remove when done - fudge for now to spead up testing
        self.psf_lookup *= 100

        self.temp_burst_dict = {}  # {"Burst #": [event # in this burst]}, for internal use

        # used in plotting paramerters
        self.plotting_colors = (["b", "r", "k", "g"])
        self.plotting_markers = (["+", "x", "o", "t"])

        # TODO: time cuts - how are these being handled?  With VEGAS this is before time cuts are applied

    def plot_psf_2dhist(self):
        '''
        Plot the psf histograms for debugging/checking
        :return:
        '''
        fig, ax = plt.subplots(2, 2)

        X, Y = np.meshgrid(self.energy_grid_tev, self.elevation_grid_deg)
        im1 = ax[0][0].pcolor(X, Y, self.psf_lookup.T)
        ax[0][0].set_xscale('log')
        ax[0][0].set_xlabel('Energy [TeV]')
        ax[0][0].set_ylabel('Elevation [deg]')

        divider = make_axes_locatable(ax[0][0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        # TODO: when interpolation is added properly plot that as well here in ax[1][0]

        e_means = (self.energy_grid_tev[:-1] + self.energy_grid_tev[1:]) / 2.
        el_means = (self.elevation_grid_deg[:-1] + self.elevation_grid_deg[1:]) / 2.
        self.logger.debug(e_means)
        for i, e in enumerate(e_means):
            label = "{0:.2f} TeV".format(e)
            self.logger.debug(e)
            self.logger.debug(el_means)
            self.logger.debug(self.psf_lookup[i, :])
            ax[0][1].plot(el_means, self.psf_lookup[i, :], label=label, marker="+", ls="--")
        ax[0][1].set_xlabel('Elevation [deg]')
        ax[0][1].set_ylabel('PSF [deg]')

        self.logger.debug(el_means)
        for i, e in enumerate(el_means):
            label = "{0:.2f}$^\circ$".format(e)
            ax[1][1].plot(e_means, self.psf_lookup[:, i], label=label, marker="+", ls="--")
        ax[1][1].set_xscale('log')
        ax[1][1].set_xlabel('Energy [TeV]')
        ax[1][1].set_ylabel('PSF [deg]')

        plt.tight_layout()
        fig.savefig("Plots/PSF.pdf")


    def get_tree_with_all_gamma(self):
        '''
        Load the gamma tree from a root file
        :param run_number: 
        :return: 
        '''

        try:
            self.load_gamma_tree()
            self.logger.debug("Read run number {0:d}".format(self.run_number))
        except:
            raise Exception("Can't read file with run_number {0:d}".format(self.run_number))

        self.logger.info("There are {0:d} events, {1:d} of which are gamma-like and pass cuts".format(self.N_all_events,
                                                                                                      self.N_gamma_events))

        self.logger.debug(self.photon_df)

        # determine the psf for each photon
        self.get_psf_lists()

    def get_run_summary(self):
        '''
        Load the run summary information from the VEGAS file, also loads the IRFs
        :return:
        '''
        self.load_run_summary()
        self.total_time_year = self.run_live_time.to(u.year)
        self.load_irfs()

    def scramble_times(self, copy=True, gamma_times_only=True):
        '''
        Estimate the background by scrambling the original event times.
        :param copy: backup data before scrambling?
        :param gamma_times_only: shuffle times after to gamma/hadron cuts
        '''
        if not hasattr(self, 'photon_df'):
            raise Exception("Call get_tree_with_all_gamma first...")

        # copy the df if not already copied
        if copy and not hasattr(self, 'photon_df_orig'):
            # if you want to keep the original burst_dict, this should only happen at the 1st scramble_times
            self.logger.debug("Saving original photon df")
            self.photon_df_orig = self.photon_df.copy()

        # shuffle the events, either using the times of all the events or the times for the post cut events
        if gamma_times_only:
            self.logger.debug("Shuffle among the arrival times of gamma events only")
            ts_ = self.photon_df.ts.values
            random.shuffle(ts_)
        else:  # use all events
            self.logger.debug("Shuffle among the arrival times of all events")

            # Note: If we are only using a subset of the data we only loaded that subset of the data into all_times
            random.shuffle(self.all_times)
            ts_ = self.all_times[:self.N_gamma_events]

        # update times with shuffled times
        self.logger.debug("Shuffled = {0:d}, original = {1:d}, num gamma = {2:d}".format(len(ts_),
                                                                                         len(self.photon_df['ts']),
                                                                                         self.N_gamma_events))
        self.photon_df.at[:, 'ts'] = ts_

        # re-init temp_burst_dict for counting
        self.temp_burst_dict = {}

        # sort by new times and reindex
        self.logger.debug("Sorting events based upon their new time stamps and reindexing")
        self.logger.debug(self.photon_df)
        if pd.__version__ > '0.18':
            self.photon_df = self.photon_df.sort_values('ts')
        else:
            self.photon_df = self.photon_df.sort('ts')

        self.photon_df.index = range(self.photon_df.shape[0])

        self.logger.debug("Sorted")
        self.logger.debug(self.photon_df)

    def random_times(self, copy=True, rate="cell", all_events=True):
        '''
        Estimate the background using random times calculated using a poisson dist.
        :param copy: backup data before scrambling?
        :param rate: method for calculating the background times (avg or cell) - only cell implemented
        :param all_events: use all events or only those passing gamma/hadron cuts
        '''
        if not hasattr(self, 'photon_df'):
            raise Exception("Call get_tree_with_all_gamma first...")

        # if you want to keep the original burst_dict, this should only happen at the 1st scramble_times
        if copy and (not hasattr(self, 'photon_df_orig')):
            self.logger.debug("Saving original photon df")
            self.photon_df_orig = self.photon_df.copy()

        # at the moment this is only defined for rate == "cell" yet the default is "avg"
        if rate == "cell":
            delta_ts = np.diff(self.photon_df.ts)
        else:
            raise Exception("Rate not define for type {0:s}".format(rate))

        if all_events:
            N = self.N_all_events
            self.rando_all_times = np.zeros(N)
            rate_expected = N * 1.0 / (self.all_times[-1] - self.all_times[0])
        else:
            N = self.photon_df.shape[0]
            rate_expected = N * 1.0 / (self.photon_df.ts.values[-1] - self.photon_df.ts.values[0])
        self.logger.info("Mean expected rate is %.2f" % rate_expected)

        for i in range(N - 1):
            if rate == "cell":
                rate_expected = 1. / delta_ts[i]
            # elif rate=="avg":
            if all_events:
                _rando_delta_t = np.random.exponential(1. / rate_expected)
                inf_loop_preventer = 0
                inf_loop_bound = 100
                while _rando_delta_t < self.veritas_deadtime_ms:
                    _rando_delta_t = np.random.exponential(1. / rate_expected)
                    inf_loop_preventer += 1
                    if inf_loop_preventer > inf_loop_bound:
                        print("Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,")
                        print("you'd better check your time unit or something...")
                self.rando_all_times[i + 1] = self.rando_all_times[i] + _rando_delta_t
            else:
                # draw a rando!
                _rando_delta_t = np.random.exponential(1. / rate_expected)
                inf_loop_preventer = 0
                inf_loop_bound = 100
                while _rando_delta_t < self.veritas_deadtime_ms:
                    _rando_delta_t = np.random.exponential(1. / rate_expected)
                    inf_loop_preventer += 1
                    if inf_loop_preventer > inf_loop_bound:
                        print("Tried 100 times and can't draw a rando wait time that's larger than VERITAS deadtime,")
                        print("you'd better check your time unit or something...")
                self.photon_df.at[i + 1, 'ts'] = self.photon_df.ts[i] + _rando_delta_t
        if all_events:
            random.shuffle(self.rando_all_times)
            for i, in range(self.photon_df.shape[0]):
                self.photon_df.at[i, 'ts'] = self.rando_all_times[i]
        # naturally sorted
        # re-init temp_burst_dict for counting
        self.temp_burst_dict = {}



    def get_psf(self, E=0.1, EL=80):
        '''
        Calculate the bin in energy and elevation and from that get the PSF for that bin
        Note: no interpolation is done currently
        :param E: energy (assumed in TeV but not checked
        :param EL: elevation in deg
        :return: psf
        '''
        energy_bin = np.digitize(E, self.energy_grid_tev, right=True) - 1
        elevation_bin = np.digitize(EL, self.elevation_grid_deg, right=True) - 1
        # TODO: add interpolation here - it must be better!!
        return self.psf_lookup[energy_bin, elevation_bin]

    def get_psf_lists(self):
        '''
        Iterate over the list of elevations and energies of gamma like events and calculate their PSF
        :return:
        '''
        if not hasattr(self, 'photon_df'):
            raise Exception("Call get_tree_with_all_gamma first...")

        # calculate psfs for gamma like events
        self.photon_df.psfs = self.get_psf(E=self.photon_df.Es.values, EL=self.photon_df.ELs.values)
        self.logger.debug("Calculated PSFs")
        self.logger.debug(self.photon_df)

    def minus_centroid_log_likelihood(self, cent_coord, coords, psfs):
        '''
        Calculate -1.*ln(L) of the centroid position
        Inputs cannot be astropy coords as used in minimizer, however, we convert internally for ease
        :param cent_coord: coordinates of the centroid [ra, dec]
        :param coords: coordinates of the events [[ras], [decs]]
        :param psfs: psfs of the events
        :return: -2.*ln(L)
        '''
        # calculate the separations in deg from the centroid
        cent_coord = SkyCoord(cent_coord[0]*u.deg, cent_coord[1]*u.deg)
        coords = SkyCoord(coords[0]*u.deg, coords[1]*u.deg)

        # separation returns an absolute distance
        thetas = cent_coord.separation(coords).deg

        # TODO: is this the same psf function as was used earlier - assume so but want to check
        ll = -2. * np.sum(np.log(psfs)) + np.sum(np.log(1. / np.cosh(np.sqrt(thetas) / psfs)))
        ll += psfs.shape[0] * np.log(1.7142 / np.pi)

        #TODO: check that the earlier cuts are using the LL not the TS or ...
        # Normalized by the number of events and multiply by minus 1 so we can get the min not the max
        # factor of 2 comes because that was here already though I suspect it shouldn't be but then the
        # other answers make no sense
        return -2. * ll / psfs.shape[0]

    def find_optimum_centroid(self, coords, psfs):
        '''
        Find the centroid with the maximum log likelihood
        :param coords:
        :param psfs:
        :return:
        '''
        init_centroid = SkyCoord(np.mean(coords.ra), np.mean(coords.dec))
        self.logger.debug("Initial centroid {0:.2f}, {1:.2f}".format(init_centroid.ra.deg,init_centroid.dec.deg))

        # minimize not maximize since that is a lot easier!
        # note that this requires that we minimize the minus log likelihood
        results = minimize(self.minus_centroid_log_likelihood,
                           [init_centroid.ra.deg, init_centroid.dec.deg],
                           args=([coords.ra.deg, coords.dec.deg], psfs),
                           method='L-BFGS-B')

        centroid = SkyCoord(results.x[0]*u.deg, results.x[1]*u.deg)
        self.logger.debug("Optimal centroid {0:.2f}, {1:.2f}".format(centroid.ra.deg, centroid.dec.deg))

        # again, to calculate the ll_counts of the centroid we
        ll_centroid = -1. * self.minus_centroid_log_likelihood([centroid.ra.deg, centroid.dec.deg],
                                                               [coords.ra.deg, coords.dec.deg],
                                                               psfs)
        return centroid, ll_centroid

    def search_angular_window(self, coords, psfs, slice_index):
        '''
        Determine if N_evt = len(coords) events are accepted to come from one direction
        slice_index is the numpy array slice of the input event numbers, used for temp_burst_dict
        return: A) centroid, likelihood, and a list of event numbers associated with this burst,
                   given that a burst is found, or the input has only one event
                B) centroid, likelihood, a list of event numbers excluding the outlier, the outlier event number
                   given that we reject the hypothesis of a burst
        :param coords: astropy.coords.SkyCoords list of coordinates
        :param psfs:
        :param slice_index:
        :return:
        '''

        num_in_slice = len(slice_index)

        assert len(coords) == num_in_slice, \
            "coords " + len(coords) + " and slice_index  " + num_in_slice + "lengths are different"

        if num_in_slice == 0:
            self.logger.debug("Slice_index is empty")
            return None, None, None, None
        elif num_in_slice == 1:
            self.logger.debug("Single event")
            return coords, 1, np.array([1])

        # find centroid
        centroid, ll_centroid = self.find_optimum_centroid(coords, psfs)
        self.logger.debug("Centroid ({0:.2f}, {1:.2f}) @ LL {2:.2f}".format(centroid.ra.deg,
                                                                            centroid.ra.deg,
                                                                            ll_centroid))

        # get ll_counts cut for given number of events in slice
        if ll_centroid <= self.ll_cut_dict.get(num_in_slice, self.ll_cut):
            # likelihood passes cut
            # all events with slice_index form a burst candidate
            return centroid, ll_centroid, slice_index
        else:
            # if not accepted, find the worst offender and
            # return the better N_evt-1 events and the outlier event
            dists = centroid.separation(coords).deg
            outlier_index = np.argmax(dists)
            mask = np.ones(len(dists), dtype=bool)
            mask[outlier_index] = False
            return centroid, ll_centroid, slice_index[mask], slice_index[outlier_index]

    def search_time_window(self, window_size=1):
        """
        Start a burst search for the given window_size in photon_df
        temp_burst_dict needs to be clean for a new scramble_times
        :param window_size: in the unit of second
        :return: burst_hist, in the process 1) fill self.temp_burst_dict, and
                                            2) fill self.photon_df.burst_sizes through burst counting; and
                                            3) fill self.photon_df.burst_sizes
        """
        assert hasattr(self, 'photon_df'), \
            "photon_df doesn't exist, read data first (read_photon_list or get_tree_with_all_gamma)"

        # check status of burst dictionary
        if len(self.temp_burst_dict) != 0:
            self.logger.info(
                "You started a burst search while there are already things in temp_burst_dict, now make it empty")
            self.temp_burst_dict = {}

        previous_window_start = -1.0
        previous_window_end = -1.0
        previous_singlets = np.array([])
        previous_non_singlets = np.array([])

        # Master event loop
        # search for events in a window after a each event
        for window_start in self.photon_df.ts:
            window_end = window_start + window_size
            self.logger.debug("Searching within window {0:.5f} to {1:.5f}".format(window_start, window_end))
            if previous_window_start == -1.0:
                # first event:
                previous_window_start = window_start
                previous_window_end = window_end
            else:
                # below just see if there are new events in the extra time interval after the previous_window_end
                new_event_slice_tuple = np.where(
                    (self.photon_df.ts >= previous_window_end) & (self.photon_df.ts < (window_end)))
                # self.logger.debug("There are {0:d} events in the time slice {1:.2f} to {2:.2f}".format(
                #     new_event_slice_tuple, t, t+window_size))
                previous_window_start = window_start
                previous_window_end = window_end
                if len(new_event_slice_tuple[0]) == 0:
                    # no new events in the extra window, continue
                    self.logger.debug("no new events found, skipping to next event")
                    continue

            # 1. slice events between t and t+window_size
            time_slice_indices = np.where((self.photon_df.ts >= window_start) & (self.photon_df.ts < (window_end)))

            # 2. remove singlets
            time_slice_indices, singlet_time_slice_indices = self.singlet_remover(np.array(time_slice_indices[0]))

            if (time_slice_indices is None) or (len(time_slice_indices) == 0):
                self.logger.debug("All events are singlet in this time window")
                continue

            # check if all new events are singlets,
            # if so the current better_events list should be contained in the previous one.
            if np.in1d(time_slice_indices, previous_non_singlets).all():
                self.logger.debug("All new events are singlets.")
                previous_singlets = singlet_time_slice_indices
                previous_non_singlets = time_slice_indices
                continue
            previous_singlets = singlet_time_slice_indices
            previous_non_singlets = time_slice_indices

            # time_slice_indices = tuple(time_slice_indices[:, np.newaxis].T)

            # _N = self.photon_df.ts.values[time_slice_indices].shape[0]
            num_in_time_slice = len(time_slice_indices)
            self.logger.debug("There are {0:d} events in the time slice".format(num_in_time_slice))
            # sanity check
            if num_in_time_slice < 1:
                self.logger.debug("All events are singlet, removed all")
                continue
            elif num_in_time_slice == 1:
                # a sparse window
                # self.photon_df.burst_sizes[slice_index] = 1
                # print "L367", slice_index
                # self.photon_df.at[slice_index[0], 'burst_sizes'] = 1
                self.logger.debug("Can't reach here")
                continue

            # 3. burst searching (in angular window)
            burst_events, outlier_events = self.search_event_slice(time_slice_indices)
            if outlier_events is None:
                # All events of slice_index form a burst, no outliers; or all events are singlet
                self.logger.debug("All events come from a single burst (or are singlets) - no outliers")
                continue
            # elif len(outlier_events)==1:
            elif outlier_events.shape[0] == 1:
                # A singlet outlier
                self.logger.debug("A singlet outlier was found")
                continue
            else:
                # If there is a burst of a subset of events, it's been taken care of,
                # now take care of the outlier slice (which has >1 events reaching here)
                outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(outlier_events)

                while outlier_of_outlier_events is not None:
                    ###QF
                    self.logger.debug("looping through the outliers ")
                    # loop until no outliers are left unprocessed
                    if len(outlier_of_outlier_events) <= 1:
                        # self.photon_df.burst_sizes[outlier_of_outlier_events[0]] = 1
                        outlier_of_outlier_events = None
                        break
                    else:
                        # more than 1 outliers to process,
                        # update outlier_of_outlier_events and repeat the while loop
                        outlier_burst_events, outlier_of_outlier_events = self.search_event_slice(
                            outlier_of_outlier_events)

        # the end of master event loop, self.temp_burst_dict is filled
        # now count bursts and fill self.photon_df.burst_sizes:
        self.logger.debug("Counting bursts")

        # copy the burst dictionary
        burst_dict = self.temp_burst_dict.copy()

        # initialize burst sizes
        self.photon_df.burst_sizes = 1

        self.logger.debug("Calculating burst sizes")
        # Note now self.temp_burst_dict will be cleared!!
        self.burst_counting()

        self.logger.debug("Calculated")
        self.logger.debug(self.photon_df)

        burst_hist = self.get_burst_hist()
        self.logger.debug("Found bursts: %s" % burst_hist)

        return burst_hist, burst_dict

    def singlet_remover(self, slice_index):
        '''
        Identify and remove any singlet events
        :param slice_index: a np array of events' indices in photon_df
        :return: new slice_index with singlets (no neighbors in a radius of 5*psf) removed, and a slice of singlets;
                 return None and input slice if all events are singlets
        '''
        if len(slice_index)== 1:
            # one event, singlet by definition:
            return None, slice_index

        N_ = self.photon_df.shape[0]

        coord_slice = SkyCoord(self.photon_df.RAs.values[slice_index]*u.deg,
                               self.photon_df.Decs.values[slice_index]*u.deg)

        psf_slice = self.photon_df.psfs.values[slice_index]

        # default all events are singlet
        mask_ = np.zeros_like(slice_index, dtype=bool)

        # use a dict of {event_num:neighbor_found} to avoid redundancy
        not_singlet_set = set([])
        for index, psf, coord in zip(slice_index, psf_slice, coord_slice):
            if index not in not_singlet_set: #check whether we already know it not a singlet
                psf_5 = psf * 5.0

                # TODO: there has to be a better way of doing this by e.g. only checking indices above a value
                for index2, coord2 in zip(slice_index, coord_slice):
                    self.logger.debug(
                        "({0:d}, {1:d}) = {2:.2f} ({3:.2f})".format(index, index2, coord.separation(coord2).deg,psf_5))
                    if (index != index2) and (index not in not_singlet_set):
                        if coord.separation(coord2).deg < psf_5:
                            # decide this pair isn't singlet
                            not_singlet_set.add(index)
                            not_singlet_set.add(index2)
                            mask_[np.where(slice_index == index)] = True
                            mask_[np.where(slice_index == index2)] = True
                            continue

        self.logger.debug("Found {0:d} singlet, {1:d} good events".format(sum(mask_ == False),
                                                                            slice_index[mask_].shape[0]))
        # this is a slice_index of the singlets and a slice_index of non-singlets
        return slice_index[mask_], slice_index[np.invert(mask_)]

    def search_event_slice(self, time_slice_indices):
        '''

        :param time_slice_indices:
        :return:
        '''
        N_ = self.photon_df.shape[0]


        # First remove singlet
        if len(time_slice_indices) == 0:
            self.logger.debug("all singlets, no bursts, and don't need to check for outliers, go to next event")
            return None, None

        # calcuate the first estimate of the centroid
        initial_coords = SkyCoord(self.photon_df.RAs.values[time_slice_indices]*u.deg,
                                  self.photon_df.Decs.values[time_slice_indices]*u.deg)

        ang_search_res = self.search_angular_window(initial_coords,
                                                    self.photon_df.psfs.values[time_slice_indices],
                                                    time_slice_indices)
        outlier_evts = []

        if len(ang_search_res) == 3:
            # All events with slice_index form 1 burst
            _, _, burst_events = ang_search_res
            self.temp_burst_dict[len(self.temp_burst_dict) + 1] = burst_events
            # burst_events should be the same as slice_index
            return burst_events, None
        else:
            while len(ang_search_res) == 4:
                # returned 4 meaning no bursts, and the input has more than one events, shall continue
                # this loop breaks when a burst is found or only one event is left,
                # in which case return values has a length of 3
                _, _, better_event_indices, outlier_event_indices = ang_search_res
                outlier_evts.append(outlier_event_indices)
                ###QF
                better_coords = SkyCoord(self.photon_df.RAs.values[better_event_indices]*u.deg,
                                  self.photon_df.Decs.values[better_event_indices]*u.deg)
                # print "in search_event_slice, candidate coords and psfs: ", better_coords,
                # self.photon_df.psfs.values[(_better_events)]
                ang_search_res = self.search_angular_window(better_coords,
                                                            self.photon_df.psfs.values[better_event_indices],
                                                            better_event_indices)

            # Now that while loop broke, we have a good list and a bad list
            _, _, burst_events = ang_search_res
            if len(burst_events) == 1:
                # No burst in slice_index found
                # count later
                return burst_events, np.array(outlier_evts)
            else:
                # A burst with a subset of events of slice_index is found
                self.temp_burst_dict[len(self.temp_burst_dict) + 1] = burst_events
                # self.photon_df.burst_sizes[tuple(burst_events)] = len(burst_events)
                return burst_events, np.array(outlier_evts)

    # def burst_counting_recur(self):
    #     """
    #     :return: nothing but fills self.photon_df.burst_sizes, during the process self.temp_burst_dict is emptied!
    #     """
    #     # Only to be called after self.temp_burst_dict is filled
    #     # Find the largest burst
    #     largest_burst_number = max(self.temp_burst_dict, key=lambda x: len(set(self.temp_burst_dict[x])))
    #     burst_size = self.temp_burst_dict[largest_burst_number].shape[0]
    #     self.logger.debug("Largest burst number {0:d} with {1:d} events.".format(largest_burst_number, burst_size))
    #
    #     for evt in self.temp_burst_dict[largest_burst_number]:
    #         # Assign burst size to all events in the largest burst
    #         self.photon_df.at[evt, 'burst_sizes'] = burst_size
    #         # self.photon_df.burst_sizes[evt] = len(self.temp_burst_dict[largest_burst_number])
    #         for key in self.temp_burst_dict.keys():
    #             # Now delete the assigned events in all other candiate bursts to avoid double counting
    #             if evt in self.temp_burst_dict[key] and key != largest_burst_number:
    #                 # self.temp_burst_dict[key].remove(evt)
    #                 self.temp_burst_dict[key] = np.delete(self.temp_burst_dict[key], np.where(self.temp_burst_dict[key] == evt))
    #
    #     # Delete the largest burst, which is processed above
    #     self.temp_burst_dict.pop(largest_burst_number, None)
    #
    #     # repeat while there are unprocessed bursts in temp_burst_dict
    #     if len(self.temp_burst_dict) >= 1:
    #         self.burst_counting_recur()

    def burst_counting(self):
        '''

        :return:
        '''
        self.logger.debug(self.temp_burst_dict)
        while len(self.temp_burst_dict) >= 1:
            # Find the largest burst
            largest_burst_number = max(self.temp_burst_dict, key=lambda x: len(set(self.temp_burst_dict[x])))
            burst_size = self.temp_burst_dict[largest_burst_number].shape[0]
            self.logger.debug("Largest burst number {0:d} with {1:d} events.".format(largest_burst_number, burst_size))

            for evt in self.temp_burst_dict[largest_burst_number]:
                # Assign burst size to all events in the largest burst
                # Use index location rather than names to since we have sorted by time
                self.photon_df.iloc[evt, 8] = burst_size

                for key in self.temp_burst_dict.keys():
                    # Now delete the assigned events in all other candiate bursts to avoid double counting
                    if evt in self.temp_burst_dict[key] and key != largest_burst_number:
                        # self.temp_burst_dict[key].remove(evt)
                        self.temp_burst_dict[key] = np.delete(self.temp_burst_dict[key],
                                                              np.where(self.temp_burst_dict[key] == evt))

            # Delete the largest burst, which is processed above
            self.temp_burst_dict.pop(largest_burst_number, None)

    # def burst_counting_fractional(self):
    #     """
    #     :return: nothing but fills self.photon_df.burst_sizes, during the process self.temp_burst_dict is emptied!
    #     """
    #     # Only to be called after self.temp_burst_dict is filled
    #     while len(self.temp_burst_dict) >= 1:
    #         # Find the largest burst
    #         largest_burst_number = max(self.temp_burst_dict, key=lambda x: len(set(self.temp_burst_dict[x])))
    #         for evt in self.temp_burst_dict[largest_burst_number]:
    #             # Assign burst size to all events in the largest burst
    #             self.photon_df.at[evt, 'burst_sizes'] = self.temp_burst_dict[largest_burst_number].shape[0]
    #             # self.photon_df.burst_sizes[evt] = len(self.temp_burst_dict[largest_burst_number])
    #             for key in self.temp_burst_dict.keys():
    #                 # Now delete the assigned events in all other candiate bursts to avoid double counting
    #                 if evt in self.temp_burst_dict[key] and key != largest_burst_number:
    #                     # self.temp_burst_dict[key].remove(evt)
    #                     self.temp_burst_dict[key] = np.delete(self.temp_burst_dict[key], np.where(self.temp_burst_dict[key] == evt))
    #         # Delete the largest burst, which is processed above
    #         self.temp_burst_dict.pop(largest_burst_number, None)

    def get_burst_hist(self):
        '''
        Produce dictionary of number of bursts of a given size
        :return: burst_hist - dictionary {number of gamma in burst: number of times}
        '''
        burst_hist = {}
        for i in np.unique(self.photon_df.burst_sizes.values):
            burst_hist[i] = np.sum(self.photon_df.burst_sizes.values == i) / i
        return burst_hist

    def signal_burst_search(self, window_size=1):
        '''

        :param window_size:
        :return:
        '''
        self.signal_burst_hist, self.signal_burst_dict = self.search_time_window(window_size=window_size)

    def estimate_bkg_burst(self, window_size=1, method="scramble_times", copy=True, n_scramble=10, rando_method="avg",
                           gamma_times_only=False):
        '''
        Estimate the background burst probability distribution
        :param window_size: window size in s
        :param method: either scramble_times (shuffle existing times) or rando (generate fake times)
        :param copy: backup photon_df first
        :param n_scramble: number of times to sample the background data
        :param rando_method: method used in rando (cell or avg) only cell implemented
        :param all:
        :return:
        '''
        self.num_scramble = n_scramble
        self.logger.debug("Using method {0:s} and scrambling {1:d} times".format(method, self.num_scramble))

        # Note that from now on we are CHANGING the photon_df!
        self.bkg_burst_hists = []

        for _ in range(self.num_scramble):
            # scramble_times the data
            if method == "scramble_times":
                self.scramble_times(copy=copy, gamma_times_only=gamma_times_only)
            elif method == "rando":
                self.random_times(copy=copy, rate=rando_method)
            else:
                raise Exception("You have tried to use method {0:s} which is not a valid method".format(method))

            # calculate the hist and dict for background bursts within time window
            bkg_burst_hist, _ = self.search_time_window(window_size=window_size)

            # append to list
            self.bkg_burst_hists.append(bkg_burst_hist.copy())
            try:
                self.avg_bkg_hist += Counter(bkg_burst_hist)
            except AttributeError:
                self.avg_bkg_hist = Counter(bkg_burst_hist)

        # normalize avg_bkg_hist by number of scrambles
        for k in self.avg_bkg_hist.keys():
            self.avg_bkg_hist[k] = self.avg_bkg_hist[k] / self.num_scramble

        self.logger.debug("Calculated average background")
        self.logger.debug(self.avg_bkg_hist)

    def get_residual_hist(self):
        residual_dict = {}
        sig_bkg_burst_sizes = set(k for dic in [self.signal_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        # fill with zero if no burst size count
        for key_ in sig_bkg_burst_sizes:
            key_ = int(key_)
            if key_ not in self.signal_burst_hist:
                self.signal_burst_hist[key_] = 0
            if key_ not in self.avg_bkg_hist:
                self.avg_bkg_hist[key_] = 0
            residual_dict[key_] = self.signal_burst_hist[key_] - self.avg_bkg_hist[key_]
        self.residual_dict = residual_dict
        return residual_dict

    def get_accept_integral(self, integral_limit=1.5):
        # \int (g(alpha, beta))**(3./2) d(cos(theta)) in eq 8.7
        rad_ = np.arange(0, integral_limit, 0.001)
        acc_ = []
        for d_ in rad_:
            acc_.append(self.accept(d_))
        accept_int = np.trapz(np.array(acc_) ** (3. / 2) * np.sin(rad_ * np.pi / 180.), x=rad_)
        self.logger.debug("The value of the acceptance integral in eq 8.7 is %.2f" % accept_int)
        return accept_int

    def n_excess(self, rho_dot, Veff):
        # eq 8.8, or maybe more appropriately call it n_expected
        if not hasattr(self, 'total_time_year'):
            self.total_time_year = (self.run_live_time * (1. - self.DeadTimeFracOn)) / 31536000.
        # n_ex = 1.0 * rho_dot * self.total_time_year * Veff
        # because the burst likelihood cut -9.5 is at 90% CL
        n_ex = 0.9 * rho_dot * self.total_time_year * Veff
        self.logger.debug("The value of the expected number of bursts (eq 8.9) is %.2f" % n_ex)
        return n_ex

    # @autojit
    def ll_counts(self, n_on, n_off, n_expected):
        '''
        eq 8.13 without the sum
        :param n_on: number of on counts
        :param n_off: number of off counts
        :param n_expected: expected number of counts
        :return: log likelihood
        '''
        return -2. * (-1. * n_expected + n_on * np.log(n_off + n_expected))

    def get_significance(self):
        residual_dict = self.residual_dict
        significance = 0
        for b_, excess_ in residual_dict.items():
            err_excess_ = np.sqrt(self.signal_burst_hist[b_] + pow(np.sqrt(10 * self.bkg_burst_hists[b_]) / 10, 2))
            self.logger.debug("Significance for bin %d has significance %.2f" % (b_, excess_ / err_excess_))
            significance += excess_ / err_excess_
        self.logger.debug("Overall Significance is %d" % significance)
        return significance

    def get_ll(self, rho_dot, burst_size_threshold, t_window, upper_burst_size=None):
        # def get_ll(self, rho_dot, burst_size_threshold, t_window, upper_burst_size=100):
        # eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
        if upper_burst_size is None:
            all_burst_sizes = set(k for dic in [self.signal_burst_hist, self.avg_bkg_hist] for k in dic.keys())
        else:
            all_burst_sizes = range(burst_size_threshold, upper_burst_size + 1)
        ll_ = 0.0
        sum_nb = 0
        self.good_burst_sizes = []  # use this to keep burst sizes that are large enough so that $n_b > \sum_b n_{b+1}$
        # for burst_size in all_burst_sizes:
        for burst_size in np.sort(np.array(list(all_burst_sizes)))[::-1]:
            # starting from the largest burst to test whether $n_b > \sum_b n_{b+1}$
            if burst_size >= burst_size_threshold:
                Veff_ = self.V_eff(burst_size, t_window)
                n_expected_ = self.n_excess(rho_dot, Veff_)
                if burst_size not in self.signal_burst_hist:
                    self.signal_burst_hist[burst_size] = 0
                if burst_size not in self.avg_bkg_hist:
                    self.avg_bkg_hist[burst_size] = 0
                n_on_ = self.signal_burst_hist[burst_size]
                n_off_ = self.avg_bkg_hist[burst_size]
                if n_on_ < sum_nb:
                    print("reached where n_b < \sum_b n_(b+1), at b={}".format(burst_size))
                    print("Stopping")
                    break
                else:
                    self.good_burst_sizes.append(burst_size)
                ll_ += self.ll_counts(n_on_, n_off_, n_expected_)
                sum_nb += n_on_
                self.logger.debug(
                    '''Adding - 2lnL at burst size % d, for search window % .1f and
                     rate density % .1f, so far - 2l nL = % .2f''' % (burst_size, t_window, rho_dot, ll_))
        self.logger.debug('''###############################################################################
                            -2lnL above burst size %d, for search window %.1f and rate density %.1f is %.2f
                            ###############################################################################''' % (
            burst_size_threshold, t_window, rho_dot, ll_))
        return ll_

    def get_ll_vs_rho_dot(self, burst_size_thresh, t_window, rho_dots=np.arange(0., 3.e5, 100), upper_burst_size=None):
        # def get_ll_vs_rho_dot(self, burst_size_thresh, t_window, rho_dots=np.arange(0., 3.e5, 100),
        # upper_burst_size=100):
        # plot a vertical slice of Fig 8-4, for a given burst size and search window,
        # scan through rho_dot and plot -2lnL
        if not isinstance(rho_dots, np.ndarray):
            rho_dots = np.asarray(rho_dots)
        lls_ = np.zeros(rho_dots.shape[0])
        for i, rho_dot_ in enumerate(rho_dots):
            lls_[i] = self.get_ll(rho_dot_, burst_size_thresh, t_window,
                                  upper_burst_size=upper_burst_size)
        return rho_dots, lls_

    # def get_minimum_ll(self, burst_size, t_window, rho_dots=np.arange(0., 3.e5, 100), return_arrays=True,
    #                    upper_burst_size=None):
    #     # search rho_dots for the minimum -2lnL
    #     if not isinstance(rho_dots, np.ndarray):
    #         rho_dots = np.asarray(rho_dots)
    #     min_ll_ = 1.e5
    #     rho_dot_min_ll_ = -1.0
    #     if return_arrays:
    #         lls_ = np.zeros(rho_dots.shape[0])
    #     i = 0
    #     for rho_dot_ in rho_dots:
    #         ll_ = self.get_ll(rho_dot_, burst_size, t_window, upper_burst_size=upper_burst_size)
    #         if ll_ < min_ll_:
    #             min_ll_ = ll_
    #             rho_dot_min_ll_ = rho_dot_
    #         if return_arrays:
    #             lls_[i] = ll_
    #             i += 1
    #     if return_arrays:
    #         return rho_dot_min_ll_, min_ll_, rho_dots, lls_
    #     return rho_dot_min_ll_, min_ll_

    # def get_ul_rho_dot(self, rho_dots, lls_, min_ll_, margin=1.e-5):
    #     """
    #     # lls_ is the **SUM** of -2lnL from *ALL* burst sizes
    #     """
    #     ll_99 = 6.63
    #     rho_dots99 = np.interp(0.0, lls_ - min_ll_ - ll_99, rho_dots)
    #     if abs(np.interp(rho_dots99, rho_dots, lls_ - min_ll_ - ll_99)) <= margin:
    #         return rho_dots99, 0
    #     ul_99_idx = (np.abs(lls_ - min_ll_ - ll_99)).argmin()
    #     ul_99_idx_all = np.where(abs(lls_ - lls_[ul_99_idx]) < margin)
    #     if ul_99_idx_all[0].shape[0] == 0:
    #         print("Can't find 99% UL!")
    #         return None
    #         # sys.exit(1)
    #     elif ul_99_idx_all[0].shape[0] > 1:
    #         print("More than one 99% UL found, strange!")
    #         print("These are rho_dot = %s, and -2lnL = %s" % (rho_dots[ul_99_idx_all], lls_[ul_99_idx_all]))
    #         return rho_dots[ul_99_idx_all], lls_[ul_99_idx_all]
    #     else:
    #         print("99%% UL found at rho_dot = %.0f, and -2lnL = %.2f" % (rho_dots[ul_99_idx], lls_[ul_99_idx]))
    #         return rho_dots[ul_99_idx], lls_[ul_99_idx]

    # """
    # # won't work:
    # def get_minimum_ll(self, burst_size, t_window):
    #     init_rho_dot = 2.e5
    #     results = minimize(self.get_ll, init_rho_dot, args=(burst_size, t_window), method='L-BFGS-B',bounds=[(0,1.e7)])
    #     if not results.success:
    #         print("Problem finding the minimum log likelihood!! ")
    #     minimum_rho_dot = results.x
    #     minimum_ll = results.fun
    #     self.logger.debug("The minimum -2lnL is %.2f at rho_dot %.1f" % (minimum_ll, minimum_rho_dot) )
    #     return minimum_rho_dot, minimum_ll
    # """

    # def get_likelihood_dict(self):
    #     ll_dict = {}
    #     residual_dict = self.residual_dict
    #
    #     sig_bkg_burst_sizes = set(k for dic in [self.sig_burst_hist, self.avg_bkg_hist] for k in dic.keys())
    #     # fill with zero if no burst size count
    #     for key_ in sig_bkg_burst_sizes:
    #         key_ = int(key_)
    #         if key_ not in self.sig_burst_hist:
    #             self.sig_burst_hist[key_] = 0
    #         if key_ not in self.avg_bkg_hist:
    #             self.avg_bkg_hist[key_] = 0
    #         residual_dict[key_] = self.sig_burst_hist[key_] - self.avg_bkg_hist[key_]
    #     return residual_dict

    def plot_burst_hist(self, filename=None, title="Burst histogram", plt_log=True, error="Poisson"):
        if not hasattr(self, 'sig_burst_hist'):
            print("self.sig_burst_hist doesn't exist, what to plot?")
            return None
        if not hasattr(self, 'avg_bkg_hist'):
            print("self.avg_bkg_hist doesn't exist, what to plot?")
            return None

        plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(3, 1, (1, 2))

        if self.avg_bkg_hist.keys() != self.signal_burst_hist.keys():
            for key in self.avg_bkg_hist.keys():
                if key not in self.signal_burst_hist:
                    self.signal_burst_hist[key] = 0
            for key in self.signal_burst_hist.keys():
                if key not in self.avg_bkg_hist:
                    self.avg_bkg_hist[key] = 0

        if error is None:
            sig_err = np.zeros(len(self.signal_burst_hist))
            bkg_err = np.zeros(len(self.avg_bkg_hist))
        elif error == "Poisson":
            sig_err = np.sqrt([v for k, v in self.signal_burst_hist])
            bkg_err = np.array([np.sqrt(v * self.num_scramble) / self.num_scramble for k, v in self.avg_bkg_hist])
        elif error.lower() == "std":
            sig_err = np.sqrt([v for k, v in self.signal_burst_hist])
            all_bkg_burst_sizes = set(k for dic in self.bkg_burst_hists for k in dic.keys())
            bkg_err = np.zeros(sig_err.shape[0])
            for key_ in all_bkg_burst_sizes:
                key_ = float(key_)
                bkg_err[key_] = np.std(np.array([d[key_] for d in self.bkg_burst_hists if key_ in d]))
        else:
            raise Exception("Unknown error type {0:s}".format(error))

        ax1.errorbar(self.signal_burst_hist.keys()[1:], self.signal_burst_hist.values()[1:], xerr=0.5,
                     yerr=sig_err[1:], fmt='bs', capthick=0,
                     label="Data")
        ax1.errorbar(self.avg_bkg_hist.keys()[1:], self.avg_bkg_hist.values()[1:], xerr=0.5,
                     yerr=bkg_err[1:], fmt='rv', capthick=0,
                     label="Background")
        plt.title(title)
        ax1.set_ylabel("Counts")
        # plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        if plt_log:
            plt.yscale('log')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.legend(loc='best')

        # plot residual
        residual_dict = self.get_residual_hist()

        if self.avg_bkg_hist.keys() != self.signal_burst_hist.keys():
            print("Check residual error calc")
        res_err = np.sqrt(sig_err ** 2 + bkg_err ** 2)

        # plt.figure()
        ax2 = plt.subplot(3, 1, 3, sharex=ax1)
        ax2.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs',
                     capthick=0,
                     label="Residual")
        ax2.axhline(y=0, color='gray', ls='--')
        ax2.set_xlabel("Burst size")
        ax2.set_ylabel("Counts")
        # plt.ylim(0, np.max(sig_burst_hist.values())*1.2)
        # plt.yscale('log')
        plt.legend(loc='best')
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_theta2(self, theta2s=np.arange(0, 2, 0.01), psf_width=0.1, N=100, const=1, ax=None, ylog=True):
        '''

        :param theta2s:
        :param psf_width:
        :param N:
        :param const:
        :param ax:
        :param ylog:
        :return:
        '''
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
        '''

        :param coords:
        :param Es:
        :param ELs:
        :param ax:
        :param color:
        :param fov_center:
        :param fov:
        :param fov_color:
        :param cent_coords:
        :param cent_marker:
        :param cent_ms:
        :param cent_mew:
        :param cent_radius:
        :param cent_color:
        :param label:
        :param projection:
        :return:
        '''
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
                # circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False, label=label)
                ellipse = Ellipse(coor, r / np.cos(np.deg2rad(coor[1])) * 2, r * 2, color=color, fill=False,
                                  label=label)
                label_flag = True
            else:
                # circ = plt.Circle(coor, radius=self.get_psf(E_, EL_), color=color, fill=False)
                ellipse = Ellipse(coor, r / np.cos(np.deg2rad(coor[1])) * 2, r * 2, color=color, fill=False)
            # ax.add_patch(circ)
            ax.add_patch(ellipse)

        label_flag = False
        if fov is not None and fov_center is not None:
            # circ_fov = plt.Circle(fov_center, radius=fov, color=fov_color, fill=False)
            ellipse_fov = Ellipse(fov_center, fov / np.cos(np.deg2rad(fov_center[1])) * 2, fov * 2, color=fov_color,
                                  fill=False)
            # ax.add_patch(circ_fov)
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
