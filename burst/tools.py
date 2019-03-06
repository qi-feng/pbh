import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def ll(n_on, n_off, n_expected):
    # eq 8.13 without the sum
    return -2. * (-1. * n_expected + n_on * np.log(n_off + n_expected))


def calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0):
    # eq 8.13, get -2lnL sum above the given burst_size_threshold, for the given search window and rho_dot
    n_expected = 0.9 * rho_dot * total_time_year * Veff
    return ll(n_on, n_off, n_expected)


def sum_ll00(pbh, rho_dot, total_time_year=None, window_sizes=[1], burst_sizes=range(2, 11), lss=['-', '--', ':'],
             cs=['r', 'b', 'k'],
             draw_grid=True, filename="ll.png", verbose=True):
    for i, window_ in enumerate(window_sizes):
        n_exps = np.zeros_like(burst_sizes).astype('float64')
        lls = np.zeros_like(burst_sizes).astype('float64')
        if total_time_year is None:
            total_time_year = pbh.total_time_year
        Veffs = []
        for j, b in enumerate(burst_sizes):
            Veff = pbh.V_eff(b, window_)
            # print("total_time_year = %.5f, rho_dot = %.4f, Veff = %.8f" % \
            #      (total_time_year, rho_dot, Veff))
            n_expected = 0.9 * rho_dot * total_time_year * Veff
            n_exps[j] = n_expected
            # print("n_expected=%.10f ll=%.10f" % (n_expected, ll(0, 0, n_expected)))
            Veffs.append(Veff)
            lls[j] = float(calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0))
            # lls[j] = ll(0, 0, n_expected)
            if verbose:
                print("Burst size %d" % b)
                print("n_expected=%.10f ll=%.10f" % (
                    n_expected, calc_ll(total_time_year, rho_dot, Veff, n_on=0, n_off=0)))
                print("log likelihood for burst size %d, 0 ON and 0 OFF is %.5f" % (b, lls[j]))
        plt.plot(burst_sizes, lls, color=cs[i], ls=lss[i], label=("search window %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    # plt.yscale('log')
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
        return a * np.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    x = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    y = n
    popt, pcov = curve_fit(gaus, x, y, p0=(-10, np.average(x, weights=n), 0.2))
    print("Fit results {}".format(popt))
    print("covariance matrix {}".format(pcov))

    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = gaus(x_fit, *popt)
    # returns x, y for plotting, and mean and sigma from fit
    return x_fit, y_fit, popt[1], popt[2], np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2]


def get_n_expected(pbh, t_window, r, verbose=False):
    # get expected gamma-ray rate from a pbh signal at distance r (pc)
    # eq 8.3; no acceptance, assume an on-axis event
    I_Value = pbh.get_integral_expected(pbh.kT_BH(t_window))
    I_Value = I_Value ** (2. / 3.) / (4. * np.pi * r * r)
    if verbose:
        print("The expected gamma-ray rate from a pbh signal at distance %.4f (pc) is %.2f" % (r, I_Value))
    return I_Value


def comp_pbhs(pbhs1, pbhs2):
    if pbhs1.total_time_year == pbhs2.total_time_year:
        print("Same total time")
    else:
        print("*** Different total time! ***")
    if pbhs1.effective_volumes == pbhs2.effective_volumes:
        print("Same effective volumes")
    else:
        print("*** Different effective volumes! ***")
    if pbhs1.get_all_burst_sizes() == pbhs2.get_all_burst_sizes():
        print("Same burst sizes")
    else:
        print("*** Different burst sizes! ***")
    if pbhs1.sig_burst_hist == pbhs2.sig_burst_hist:
        print("Same sig_burst_hist")
    else:
        print("*** Different sig_burst_hist! ***")
        print("{0}, {1}".format(pbhs1.sig_burst_hist, pbhs2.sig_burst_hist))
    if pbhs1.avg_bkg_hist == pbhs2.avg_bkg_hist:
        print("Same avg_bkg_hist")
    else:
        print("*** Different avg_bkg_hist! ***")
        print("{0}, {1}".format(pbhs1.avg_bkg_hist, pbhs2.avg_bkg_hist))
    if pbhs1.window_size == pbhs2.window_size:
        print("Same window_size")
    else:
        print("*** Different window_size! ***")
    if pbhs1.get_ULs() == pbhs2.get_ULs():
        print("Same UL for burst size threshold 2")
    else:
        print("*** Different ULs for burst size threshold 2! ***")
