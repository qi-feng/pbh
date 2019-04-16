import numpy as np
from matplotlib import pyplot as plt

from main.pbh import Pbh
from main.powerlaw import powerlaw


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


def sim_psf_likelihood_scramble_data(Nsim=1000, N_burst=3,
                                     runNum=55480,
                                     filename=None,
                                     sig_bins=50, bkg_bins=100, ylog=True,
                                     ):
    # EL = 75, fov_center = np.array([180., 30.0])
    fov = 1.75
    pbh = Pbh()
    pbh.get_tree_with_all_gamma(run_number=runNum, nlines=None)

    if Nsim >= pbh.photon_df.shape[0] - 1:
        print(
            "Only {} events, doing {} sims instead of {}...".format(pbh.photon_df.shape[0], pbh.photon_df.shape[0] - 1,
                                                                    Nsim))
        Nsim = pbh.photon_df.shape[0] - 1
    # Nsim = 1000
    ll_bkg_all = np.zeros(Nsim)
    ll_sig_all = np.zeros(Nsim)

    for j in range(Nsim):
        pbh.scramble_times()

        #
        # this_slice = pbh.photon_df.iloc[j*N_burst:(j+1)*N_burst]
        this_slice = pbh.photon_df.iloc[j:j + N_burst]
        rand_Es = this_slice.Es.values

        # rand_bkg_coords = np.zeros((N_burst, 2))
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


def sim_cut_90efficiency(NMC=50, Nsim=2000, N_burst=3, upper=True, plot=True, outfile=None,
                         EL=75, fov_center=np.array([180., 30.0]),
                         cut_bins=50, ylog=False):
    cuts = np.zeros(NMC).astype('float')
    for trial in range(NMC):
        ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=None,
                                                    EL=EL, fov_center=fov_center,
                                                    sig_bins=50, bkg_bins=100, ylog=True)
        ll_cuts = np.arange(-15, 40, 0.05)
        label = "Burst size " + str(N_burst) + ", Dec {:.0f}".format(fov_center[1])
        # for i, c in enumerate(ll_cuts):
        #    sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
        #    sigs[i] = sig
        best_cut = np.percentile(ll_sig_all, 90)
        cuts[trial] = best_cut
    if plot:
        plt.hist(cuts, bins=cut_bins, color='r', alpha=0.3, label=str(label) + " sim cuts")
        # plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut {:.2f}".format(best_cut))
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
        ll_cuts = np.arange(-15, 40, 0.05)
        label = "Burst size " + str(N_burst) + ", Run {}".format(runNum)
        # for i, c in enumerate(ll_cuts):
        #    sig = calc_cut_sig(ll_sig_all, ll_bkg_all, c, upper=upper)
        #    sigs[i] = sig
        best_cut = np.percentile(ll_sig_all, 90)
        cuts[trial] = best_cut
    if plot:
        plt.hist(cuts, bins=cut_bins, color='r', alpha=0.3, label=str(label) + " sim cuts")
        # plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut {:.2f}".format(best_cut))
        plt.legend(loc='best')
        plt.xlabel("Likelihood")
        if ylog:
            plt.yscale('log')
        if outfile is not None:
            plt.savefig(outfile)
        plt.show()
    return cuts
