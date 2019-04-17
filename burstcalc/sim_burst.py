import numpy as np
from scipy import integrate

from burstcalc.burst import BurstFile


class SimBurst(BurstFile):
    '''
    A burst file instance with all of the required functions for a simulated search

    '''

    def __init__(self, run_number, data_dir, using_ed=True, num_ev_to_read=None, debug=False,
                 veritas_deadtime_ms=0.33e-3):

        super().__init__(run_number=run_number, data_dir=data_dir, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                         debug=debug, veritas_deadtime_ms=veritas_deadtime_ms)

    def gen_one_random_coords_projected_plane(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        # the dec of this small circle should be in the range of [cent_dec - theta, cent_dec + theta]
        delta_dec = (np.random.random() * 2. - 1.) * theta
        _dec = cent_coord[1] + delta_dec
        # Note that dec is 90 deg - theta in spherical coordinates
        _ra = cent_coord[0] + np.rad2deg(np.arccos(
            np.cos(np.deg2rad(theta)) * (1. / np.cos(np.deg2rad(90. - cent_coord[1]))) * (
                    1. / np.cos(np.deg2rad(90. - _dec))) \
            - np.tan(np.deg2rad(90. - cent_coord[1])) * np.tan(np.deg2rad(90. - _dec))))
        # _ra = cent_coord[0] + rad2deg( np.arccos ( np.cos(deg2rad(theta)) *
        # (1./np.cos(deg2rad(cent_coord[1]))) * (1./np.cos(deg2rad(_dec))) \
        #                                           - np.tan(deg2rad(cent_coord[1])) *  np.tan(deg2rad(_dec)) ) )
        return np.array([_ra, _dec])

    def gen_one_random_coords(self, cent_coord, theta):
        """
        *** Here use small angle approx, as it is only a sanity check ***
        :return a pair of uniformly random RA and Dec at theta deg away from the cent_coord
        """
        _phi = np.random.random() * np.pi * 2.
        # _ra = cent_coord[0] + np.sin(_phi) * theta
        _ra = cent_coord[0] + np.sin(_phi) * theta / np.cos(np.deg2rad(cent_coord[1]))
        _dec = cent_coord[1] + np.cos(_phi) * theta
        return np.array([_ra, _dec])

    def gen_one_random_theta(self, psf_width, prob="psf", fov=1.75):
        """
        :prob can be either "psf" that uses the hyper-sec function, or "uniform", or "gauss"
        """
        if prob.lower() == "psf" or prob == "hypersec" or prob == "hyper secant":
            # _rand_theta = np.random.random()*fov
            _rand_test_cdf = np.random.random()
            # _thetas = np.arange(0, fov, 0.001)
            # _theta2s = _thetas ** 2
            _theta2s = np.arange(0, fov ** 2, 1e-4)
            _thetas = np.sqrt(_theta2s)
            _psf_pdf = self.psf_func(_theta2s, psf_width, N=1)
            # _cdf = np.cumsum(_psf_pdf - np.min(_psf_pdf))
            # _cdf = integrate.cumtrapz(_psf_pdf, _thetas, initial=0)
            _cdf = integrate.cumtrapz(_psf_pdf, _theta2s, initial=0)
            _cdf = _cdf / np.max(_cdf)
            # y_interp = np.interp(x_interp, x, y)
            _theta2 = np.interp(_rand_test_cdf, _cdf, _theta2s)
            return np.sqrt(_theta2)
        elif prob.lower() == "uniform" or prob == "uni":
            return np.random.random() * fov
        # gauss may have a caveat as this is not important
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

    def psf_func(self, theta2, psf_width, N=100):
        '''
        Not sure yet
        :param theta2:
        :param psf_width:
        :param N:
        :return:
        '''
        return 1.7142 * N / 2. / np.pi / (psf_width ** 2) / np.cosh(np.sqrt(theta2) / psf_width)
        # equivalently:
        # return (stats.hypsecant.pdf(np.sqrt(theta2s)/psf_width)*1.7142/2./psf_width**2)

    def psf_cdf(self, psf_width, fov=1.75):
        '''
        Not sure yet
        :param psf_width: same as psf_func
        :param fov: given so that we calculate cdf from 0 to fov
        :return:
        '''

        _thetas = np.arange(0, fov, 0.001)
        _theta2s = _thetas ** 2
        # cdf = np.cumsum(self.psf_func(theta2s, psf_width, N=1))
        cdf = integrate.cumtrapz(self.psf_func(_theta2s, psf_width, N=1), _thetas, initial=0)
        cdf = cdf / np.max(cdf)
        return cdf
