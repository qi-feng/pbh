import numpy as np

from burstcalc.burst import BurstFile


class PbhBurst(BurstFile):
    '''
    A burst file instance with all of the required functions for a pbh search

    '''

    def __init__(self, run_number, data_dir, using_ed=True, num_ev_to_read=None, debug=False,
                 veritas_deadtime_ms=0.33e-3):
        super().__init__(run_number=run_number, data_dir=data_dir, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                         debug=debug, veritas_deadtime_ms=veritas_deadtime_ms)

    def kT_BH(self, t_window):
        # eq 8.2
        # temperature (energy in the unit of GeV) of the BH
        return 7.8e3 * (t_window) ** (-1. / 3)

    def V_eff(self, burst_size, t_window):
        # eq 8.7; time is in the unit of year
        I_Value = self.get_integral_expected(self.kT_BH(t_window))
        rad_Int = self.get_accept_integral()
        effVolume = (1. / (8 * np.sqrt(np.pi))) * gamma(burst_size - 1.5) / factorial(
            burst_size) * I_Value * rad_Int  # * self.total_time_year
        self.logger.debug("The value of the effective volume (eq 8.7) is %.2f" % effVolume)
        return effVolume

    def diff_raw_number(self, E, kT_BH):
        # eq 8.1
        # The expected dN/dE of gammas at energy E (unit GeV), temp kT_BH(tau)
        if E < kT_BH:
            n_exp_gamma = 9.e35 * kT_BH ** (-1.5) * E ** (-1.5)
        else:
            n_exp_gamma = 9.e35 * E ** (-3)
        return n_exp_gamma


    def get_integral_expected(self, kT_BH):
        # integrate over EA between energies:
        self.elo = 80.
        self.ehi = 50000.
        # the integral part of eq 8.3 with no acceptance raised to 3/2 power (I^3/2 in eq 8.7);
        # EA normalized to the unit of pc^2
        # The expected # of gammas
        if not hasattr(self, 'EA'):
            self.logger.info("self.EA doesn't exist, reading it now")
            self.get_run_summary()
        # 2D array, energy and (dN/dE * EA)
        number_expected = np.zeros((self.EA.shape[0], 2))
        count = 0
        for e_, ea_ in self.EA:
            # print e_, ea_
            diff_n_exp = self.diff_raw_number(10 ** e_ * 1000., kT_BH)
            number_expected[count, 0] = 10 ** e_ * 1000.
            number_expected[count, 1] = diff_n_exp * ea_ / (3.086e+16 ** 2)
            count += 1
            # 1 pc = 3.086e+16 m
        energy_cut_indices = np.where((number_expected[:, 0] >= self.elo) & (number_expected[:, 0] <= self.ehi))
        integral_expected = np.trapz(number_expected[energy_cut_indices, 1], x=number_expected[energy_cut_indices, 0])
        # This is the "I**(3./2)" in eq 8.7 in Simon's thesis
        integral_expected = integral_expected ** (3. / 2.)
        self.logger.debug("The value of I in eq 8.7 is %.2f" % integral_expected)
        return integral_expected
