import numpy as np
from matplotlib import pyplot as plt

from main.sim import sim_psf_likelihood, sim_cut_90efficiency
from main.tools import fit_gaussian_hist


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


def opt_cut(Nsim=10000, N_burst=3):
    # optimize cuts based on Monte Carlo, maximizing significance
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None, sig_bins=50, bkg_bins=100,
                                                ylog=True)
    ll_cuts = np.arange(-15, 40, 0.1)
    sigs, best_cut = cut_optimize(ll_sig_all, ll_bkg_all, ll_cuts,
                                  label="Burst size " + str(N_burst),
                                  outfile="psf_likelihood_cut_optimization_sim" + str(Nsim) + "_burst_size" + str(
                                      N_burst) + ".png")
    return best_cut


def eff_cut(Nsim=10000, N_burst=3):
    # find 90% efficiency cuts
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None, sig_bins=50, bkg_bins=100,
                                                ylog=True)
    ll_cuts = np.arange(-15, 40, 0.1)
    best_cut = cut_90efficiency(ll_sig_all, ll_bkg_all, ll_cuts,
                                label="Burst size " + str(N_burst),
                                outfile="psf_likelihood_cut_90efficiency_sim" + str(Nsim) + "_burst_size" + str(
                                    N_burst) + ".png")
    return best_cut


def psf_cut_search(NMC=1000, Nsim=1000, bss=range(2, 11), fov_center=np.array([180., 80.0])):
    # cut_list = []
    best_cuts = []
    dec = fov_center[1]
    # for bs_ in range(2,10): #kernel died
    for bs_ in bss:
        sim_cuts_Dec80_ = sim_cut_90efficiency(NMC=NMC, Nsim=Nsim, N_burst=bs_,
                                               fov_center=fov_center,
                                               outfile="sim_cuts_distr_bs" + str(bs_) + "_Dec" + (
                                                   "{:.0f}".format(dec)) + "_1M_sims_v4.png")

        np.save("sim_cuts_distr_bs" + str(bs_) + "_Dec" + ("{:.0f}".format(dec)) + "_1M_sims_v4.npy", sim_cuts_Dec80_)
        # cut_list.append(sim_cuts_Dec80_)
        hist_, bins_, _ = plt.hist(sim_cuts_Dec80_, bins=50, color='r', alpha=0.3,
                                   label="burst size {}, Dec={:.0f}$^\circ$".format(bs_, dec))
        x_fit, y_fit, mu, sig, dmu, dsig = fit_gaussian_hist(bins_, hist_)
        plt.plot(x_fit, y_fit, 'r--')
        # plt.axvline(mu, color='r', ls='--', label="mean: {:.2f}$\pm${:.2f} \n sigma: {:.2f}$\pm${:.2f}".format(mu, dmu, sig, dsig))
        plt.axvline(mu, color='r', ls='--',
                    label="mean: {:.3f}$\pm${:.3f} \n sigma: {:.3f}$\pm${:.3f}".format(mu, dmu, sig, dsig))
        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        plt.savefig("sim_cuts_distr_bs" + str(bs_) + "_Dec" + ("{:.0f}".format(dec)) + "_1M_sims_fit_v4.pdf")
        best_cuts.append(mu)
        # plt.show()
    print("best cuts: {}")
    return best_cuts
