import numpy as np


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
