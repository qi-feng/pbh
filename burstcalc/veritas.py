import os

import numpy as np
import pandas as pd

from astropy import units as u

import logging

try:
    import ROOT

    ROOT.PyConfig.StartGuiThread = False
except:
    raise Exception("Can't import ROOT, no related functionality possible")


class VeritasFile(object):
    def __init__(self, run_number, data_dir,  using_ed=True, num_ev_to_read=None, debug = False):
        self.logger = logging.getLogger(__name__)

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.run_number = int(run_number)
        self.data_dir = data_dir

        self.logger.debug('Run Number = %d ', self.run_number)
        self.logger.debug('Data Dir = %s ', self.data_dir)

        if using_ed:
            self.logger.info('Loading ED file')

            self.using_ed = True
            self._read_ed_file()
        else:
            self.logger.info('Loading VEGAS file')
            self.using_ed = False
            self.__vegas_loaded = False

            if (not self.__vegas_loaded):
                self.logger.debug('Load VEGAS lib')
                if ROOT.gSystem.Load("libVEGAScommon") not in [0, 1]:
                    raise Exception("Problem loading VEGAS Common libraries - please check this before proceeding")
                if ROOT.gSystem.Load("libVEGASstage4") not in [0, 1]:
                    raise Exception("Problem loading VEGAS Stage 4 libraries - please check this before proceeding")
                if ROOT.gSystem.Load("libVEGASstage5") not in [0, 1]:
                    raise Exception("Problem loading VEGAS Stage 5 libraries - please check this before proceeding")
                if ROOT.gSystem.Load("libVEGASstage6") not in [0, 1]:
                    raise Exception("Problem loading VEGAS Stage 6 libraries - please check this before proceeding")
                self.__vegas_loaded = True
            self._read_vegas_file()

        self.file_is_loaded = False

        # pandas df columns
        # MJDs = MJD day (int)
        # ts = time of day [ns]
        # RAs = RA [deg]
        # DECs = Dec [deg]
        # Es = energy [TeV]
        # ELs = elevations [deg]
        # psfs - not filled here
        # burst_sizes - not filled here
        # fail_cut - not filled here
        self.columns = ['MJDs', 'ts', 'RAs', 'Decs', 'Es', 'ELs', 'psfs', 'burst_sizes', 'fail_cut']

        # number of events to read (for faster debugging)
        self.num_ev_to_read = num_ev_to_read

    def set_cuts(self, cuts="soft", energy_cut_low = 0.08*u.TeV, energy_cut_high = 50*u.TeV, elevation_cut_low = 50, distance_upper_cut = 1.45):
        '''

        :param cuts:
        :return:
        '''
        self.apply_cuts = True
        self.cuts = cuts
        self.energy_low_cut = energy_cut_low
        self.energy_high_cut = energy_cut_high
        self.elevation_low_cut = elevation_cut_low
        self.distance_upper_cut = distance_upper_cut

        if self.cuts == "soft":
            self.MSW_lower = 0.05
            self.MSW_upper = 1.1
            self.MSL_lower = 0.05
            self.MSL_upper = 1.3
            self.max_height_lower = 7

        self.cuts_set = True


    def _read_ed_file(self):
        '''
        Read in an event display file
        :return: root TFile
        '''
        self.filename = str(self.data_dir) + "/" + str(self.run_number) + ".anasum.root"
        if os.path.isfile(self.filename):
            self.root_file = ROOT.TFile(self.filename, "read")
            self.file_is_loaded = True
        else:
            raise Exception("Unable to load file {0:s}".format(self.filename))

    def _read_vegas_file(self):
        '''
        Read in a VEGAS file.
        File name assumed to be of form [data_dir]/[run_number].St[5/6].root
        Note: we are using stage 4 and 6 files as they have all of the detail that we need without the baggage
        we need to produce one stage 6 file per run.  This file needs to be run producing an upper limit since
        the effective area is extacted from the upper limit object
        :param run_number: run number
        :param data_dir: data directory
        :return: VAROOTIO
        '''
        self.filename = str(self.data_dir) + "/" + str(self.run_number) + ".St6.root"
        if os.path.isfile(self.filename):
            self.root_file = ROOT.TFile(self.filename, "read")
        else:
            raise Exception("Unable to load file {0:s}".format(self.filename))

        self.st4_filename = str(self.data_dir) + "/" + str(self.run_number) + ".St4.root"
        if os.path.isfile(self.st4_filename):
            self.st4_root_file = ROOT.VARootIO(self.st4_filename, True)
        else:
            raise Exception("Unable to load file {0:s}".format(self.filename))

        self.file_is_loaded = True

    def _load_gamma_tree_ed(self):
        '''
        Load the gamma-ray information from an ED root file
        :return:
        '''

        # check whether the root file is loaded
        if not self.file_is_loaded:
            self._read_ed_file()

        # load the gamma tree
        all_gamma_tree = self.root_file.Get("run_" + str(self.run_number) + "/stereo/TreeWithAllGamma")

        # check whether reading all events
        if self.num_ev_to_read is not None:
            self.N_all_events = self.num_ev_to_read
        else:
            self.N_all_events = all_gamma_tree.GetEntries()

        # create dataframe
        self.df_ = pd.DataFrame(np.array([np.zeros(self.N_all_events)] * len(self.columns)).T,
                                columns=self.columns)

        # note we want to record all of the event times and the number of gamma events
        self.all_times = np.zeros(self.N_all_events)
        self.N_gamma_events = 0

        # perform main loop to fill tree
        for i, event in enumerate(all_gamma_tree):
            self.all_times[i] = event.timeOfDay
            distance = np.sqrt(event.Xderot * event.Xderot + event.Yderot * event.Yderot)

            # test if a gamma event
            if (event.Energy < self.energy_low_cut) or (event.Energy > self.energy_high_cut) or (
                    event.TelElevation < self.elevation_low_cut) or (
                    event.IsGamma == 0) or (distance > self.distance_upper_cut):
                self.df_.fail_cut.at[i] = 1
            else:
                self.N_gamma_events += 1
                # fill the pandas dataframe
                self.df_.MJDs[i] = event.dayMJD
                # df_.eventNumber[i] = event.eventNumber
                self.df_.ts[i] = event.timeOfDay
                self.df_.RAs[i] = event.GammaRA
                self.df_.Decs[i] = event.GammaDEC
                self.df_.Es[i] = event.Energy
                self.df_.ELs[i] = event.TelElevation

    def _load_gamma_tree_vegas(self):
        '''
        Load the gamma event information from the VEGAS st4 root tree
        :return:
        '''
        if not self.file_is_loaded:
            self._read_vegas_file()

        # load the event tree
        event_tree = self.st4_root_file.loadTheShowerEventTree()

        # check whether reading all events
        if self.num_ev_to_read is not None:
            self.N_all_events = self.num_ev_to_read
        else:
            self.N_all_events = event_tree.GetEntries()

        # create dataframe
        self.df_ = pd.DataFrame(np.array([np.zeros(self.N_all_events)] * len(self.columns)).T,
                                columns=self.columns)

        # note we want to record all of the event times and the number of gamma events
        self.all_times = np.zeros(self.N_all_events)
        self.N_gamma_events = 0

        # perform main loop to fill tree
        for i, event in enumerate(event_tree):
            self.logger.debug(i)
            if i >= self.N_all_events:
                break
            # distance in deg from center of FoV
            distance = np.sqrt(event.S.fDirectionXCamPlane_Deg ** 2 +
                               event.S.fDirectionYCamPlane_Deg ** 2)

            # this is the time of day in ns
            self.all_times[i] = event.S.fTime.getDayNS()

            if self.cuts == "soft":
                is_gamma = ((event.S.fMSW > self.MSW_lower) and (event.S.fMSW < self.MSW_upper) and
                            (event.S.fMSL > self.MSL_lower) and (event.S.fMSW < self.MSL_upper) and
                            (event.S.fShowerMaxHeight_KM > self.max_height_lower))
                self.logger.debug("{0:0.2f} {1:0.2f} {2:0.2f} {3:s}".format(event.S.fMSW, event.S.fMSL, event.S.fShowerMaxHeight_KM, str(is_gamma)))
            else:
                is_gamma = True

            # test if a gamma event
            if (event.S.fEnergy_GeV*u.GeV < self.energy_low_cut) or (event.S.fEnergy_GeV*u.GeV > self.energy_high_cut) or (
                    event.S.fArrayTrackingElevation_Deg < self.elevation_low_cut) or (
                    distance > self.distance_upper_cut) or (not is_gamma):
                self.df_.fail_cut.at[i] = 1
            else:
                self.N_gamma_events += 1
                # fill the pandas dataframe
                self.df_.MJDs[i] = event.S.fTime.getMJDInt()
                self.df_.ts[i] = event.S.fTime.getDayNS()
                self.df_.RAs[i] = np.rad2deg(event.S.fDirectionRA_J2000_Rad)
                self.df_.Decs[i] = np.rad2deg(event.S.fDirectionDec_J2000_Rad)
                self.df_.Es[i] = event.S.fEnergy_GeV / 1000. # to TeV
                self.df_.ELs[i] = event.S.fArrayTrackingElevation_Deg


    def load_gamma_tree(self):
        '''
        Lod gamma ray data from ED/VEGAS file
        :return:
        '''
        if not self.cuts_set:
            Exception("Cuts have not been set - this mudt be done before loading the gamma data")

        if self.using_ed:
            return self._load_gamma_tree_ed()
        else:
            return self._load_gamma_tree_vegas()

    def _load_run_summary_ed(self):
        '''
        Load the details of the run (target, number of counts etc) from ED
        :return:
        '''
        if not self.file_is_loaded:
            self._read_ed_file()

        tRunSummary = self.root_file.Get("total_1/stereo/tRunSummary")
        for tR in tRunSummary:
            self.run_duration = tR.tOn * u.s
            self.DeadTimeFracOn = tR.DeadTimeFracOn
            self.run_live_time = self.run_live_time * (1. - self.DeadTimeFracOn)
            self.TargetRA = tR.TargetRA
            self.TargetDec = tR.TargetDec
            self.TargetRAJ2000 = tR.TargetRAJ2000
            self.TargetDecJ2000 = tR.TargetDecJ2000
            break

    def _load_run_summary_vegas(self):
        '''
        load the VEGAS run summary from the root file
        :return:
        '''
        if not self.file_is_loaded:
            self._read_vegas_file()

        run_stats_tree = self.root_file.Get("RunStatsTree")
        for tR in run_stats_tree:
            if tR.faRunNumber != self.run_number:
                raise Exception(
                    "Run number in file ({0:d}) is not the same as run number of file ({1:d})".format(tR.faRunNumber,
                                                                                                      self.run_number))
            self.run_duration = tR.faDuration * u.s
            self.run_live_time = tR.faLiveTime * u.s
            self.DeadTimeFracOn = 1. - self.run_live_time / self.run_duration
            self.TargetRA = tR.fSourceRADeg
            self.TargetDec = tR.fSourceDecDeg
            self.TargetRAJ2000 = tR.fSourceRADeg
            self.TargetDecJ2000 = tR.fSourceDecDeg
            break

    def load_run_summary(self):
        '''
        Load the run summary from the root file for the given run
        :param self: a pbh class instance to be loaded into
        :return:
        '''
        if self.using_ed:
            self._load_run_summary_ed()
        else:
            self._load_run_summary_vegas()

    def _load_irfs_ed(self):
        '''
        Load event display IRFS
        :param self:
        :return:
        '''
        ea = self.root_file.Get("run_" + str(self.run_number) + "/stereo/EffectiveAreas/gMeanEffectiveArea")
        self.EA = np.zeros((ea.GetN(), 2))
        for i in range(ea.GetN()):
            self.EA[i, 0] = ea.GetX()[i]
            self.EA[i, 1] = ea.GetY()[i]

        self.accept = self.root_file.Get("run_" + str(self.run_number) + "/stereo/RadialAcceptances/fAccZe_0")
        # use self.accept.Eval(x) to get y, or just self.accept(x)

    def _load_irfs_vegas(self):
        ea = self.root_file.Get("UpperLimit/VAUpperLimit").GetEffectiveArea()
        self.EA = np.zeros((ea.GetN(), 2))
        for i in range(ea.GetN()):
            self.EA[i, 0] = ea.GetX()[i]
            self.EA[i, 1] = ea.GetY()[i]

        self.accept = self.root_file.Get("RingBackgroundModelAnalysis/AcceptanceCurves/AcceptanceForRun{0:d}".format(self.run_number))
        print(self.accept)

    def load_irfs(self):
        if self.using_ed:
            self._load_irfs_ed()
        else:
            self._load_irfs_vegas()
