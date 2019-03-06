from matplotlib import pyplot as plt
import numpy as np

from burst.pbh import Pbh
from burst.powerlaw import powerlaw
from burst.sim import sim_psf_likelihood_scramble_data, sim_psf_likelihood
from burst.io import dump_pickle


def test_psf_func(Nburst=10, filename=None, cent_ms=8.0, cent_mew=1.8):
    # Nburst: Burst size to visualize
    pbh = Pbh()
    fov_center = np.array([180., 30.0])
    fov = 1.75

    # spec sim:
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
    # def test_psf_func_sim(psf_width=0.05, prob="psf", Nsim=10000, Nbins=40, filename=None, xlim=None):
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
    ll_sig_all, ll_bkg_all = sim_psf_likelihood(Nsim=Nsim, N_burst=N_burst, filename=filename, sig_bins=sig_bins,
                                                bkg_bins=bkg_bins, ylog=ylog)
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
    return None


def test_sim_likelihood_from_data(Nsim=1000, N_burst=3, runNum=55480, filename=None, sig_bins=50, bkg_bins=100,
                                  ylog=True):
    ll_sig_all, ll_bkg_all = sim_psf_likelihood_scramble_data(Nsim=Nsim, N_burst=N_burst, runNum=runNum)
    best_cut = np.percentile(ll_sig_all, 90)
    plt.hist(ll_sig_all, bins=sig_bins, color='r', alpha=0.3, label="Burst size " + str(N_burst) + " signal")
    plt.hist(ll_bkg_all, bins=bkg_bins, color='b', alpha=0.3, label="Burst size " + str(N_burst) + " background")
    plt.axvline(x=best_cut, ls="--", lw=0.3, label="90% efficiency cut = {:.2f}".format(best_cut))
    plt.legend(loc='best')
    plt.xlabel("Likelihood")
    if ylog:
        plt.yscale('log')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None


def test_sim_likelihood_from_data_all(Nsim=1000, N_bursts=range(2, 11), runNum=55480, filename=None, sig_bins=50,
                                      bkg_bins=100, ylog=True):
    fov = 1.75
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=None)

    # if Nsim >= pbh.photon_df.shape[0] - 1:
    #    print("Only {} events, doing {} sims instead of {}...".format(pbh.photon_df.shape[0], pbh.photon_df.shape[0] - 1, Nsim))
    #    Nsim = pbh.photon_df.shape[0] - 1
    # Nsim = 1000
    ll_bkg_all = np.zeros((len(N_bursts), Nsim)).astype(float)
    ll_sig_all = np.zeros((len(N_bursts), Nsim)).astype(float)
    best_cuts = np.zeros(len(N_bursts))
    fig, axes = plt.subplots(3, len(N_bursts) / 3, figsize=(18, 18))

    for xx, N_burst in enumerate(N_bursts):
        sim_counter = 0
        N_evt_segments = pbh.photon_df.shape[0] // N_burst
        while sim_counter < Nsim:
            pbh.scramble()
            for j in range(N_evt_segments):
                # pbh.scramble()
                #
                this_slice = pbh.photon_df.iloc[j * N_burst:(j + 1) * N_burst]
                # this_slice = pbh.photon_df.iloc[j:j+N_burst]
                # rand_Es = this_slice.Es.values

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
                ll_bkg_all[xx, sim_counter] = ll_bkg
                ll_sig_all[xx, sim_counter] = ll_sig
                sim_counter += 1
                if sim_counter >= Nsim:
                    break

        best_cuts[xx] = np.percentile(ll_sig_all[xx], 90)
        axes.flatten()[xx].hist(ll_sig_all[xx], bins=sig_bins, color='r', alpha=0.3,
                                label="Burst size " + str(N_burst) + " signal")
        axes.flatten()[xx].hist(ll_bkg_all[xx], bins=bkg_bins, color='b', alpha=0.3,
                                label="Burst size " + str(N_burst) + " background")
        axes.flatten()[xx].axvline(x=best_cuts[xx], ls="--", lw=0.3,
                                   label="90% efficiency cut = {:.2f}".format(best_cuts[xx]))
        axes.flatten()[xx].legend(loc='best')
        axes.flatten()[xx].set_xlabel("Likelihood")
        if ylog:
            axes.flatten()[xx].set_yscale('log')
    plt.tight_layout()
    print(best_cuts)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return None


def test_burst_finding(window_size=3, runNum=55480, nlines=None, N_scramble=3, plt_log=True, verbose=False,
                       save_hist="test_burst_finding_histo", bkg_method="scramble", rando_method="avg"):
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=nlines)
    # do a small list
    # pbh.photon_df = pbh.photon_df[:nlines]
    sig_burst_hist, sig_burst_dict = pbh.sig_burst_search(window_size=window_size, verbose=verbose)

    # avg_bkg_hist = pbh.estimate_bkg_burst(window_size=window_size, method="scramble", copy=True, n_scramble=N_scramble)
    avg_bkg_hist, bkg_burst_dicts = pbh.estimate_bkg_burst(window_size=window_size, method=bkg_method,
                                                           rando_method=rando_method,
                                                           copy=True, n_scramble=N_scramble, return_burst_dict=True,
                                                           verbose=verbose)

    dump_pickle(sig_burst_hist, save_hist + str(window_size) + "_sig_hist.pkl")
    dump_pickle(sig_burst_dict, save_hist + str(window_size) + "_sig_dict.pkl")
    dump_pickle(bkg_burst_dicts, save_hist + str(window_size) + "_bkg_dicts.pkl")

    if nlines is None:
        filename = save_hist + "_AllEvts" + "_Nscrambles" + str(N_scramble) + "_window" + str(
            window_size) + "_method_" + str(bkg_method) + ".png"
    else:
        filename = save_hist + "_Nevts" + str(nlines) + "_Nscrambles" + str(N_scramble) + "_window" + str(
            window_size) + "_method_" + str(bkg_method) + ".png"

    pbh.plot_burst_hist(filename=filename,
                        title="Burst histogram " + str(window_size) + "-s window " + str(bkg_method) + " method",
                        plt_log=True, error="Poisson")

    print("Done!")

    return pbh


def test_ll(window_sizes=[1, 2, 5, 10], colors=['k', 'r', 'b', 'magenta'], runNum=55480, N_scramble=3, verbose=False,
            rho_dots=np.arange(0., 5.e5, 100), save_hist="test_ll", bkg_method="scramble", rando_method="avg",
            burst_size=2, xlog=True, grid=True):
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=runNum, nlines=None)

    for ii, window_size in enumerate(window_sizes):
        sig_burst_hist, sig_burst_dict = pbh.sig_burst_search(window_size=window_size, verbose=verbose)
        avg_bkg_hist, bkg_burst_dicts = pbh.estimate_bkg_burst(window_size=window_size, method=bkg_method,
                                                               rando_method=rando_method,
                                                               copy=True, n_scramble=N_scramble, return_burst_dict=True,
                                                               verbose=verbose)
        # rho_dots, lls = pbh.get_ll_vs_rho_dot(burst_size, window_size, rho_dots=rho_dots, verbose=verbose)
        # minimum_rho_dot, minimum_ll = pbh.get_minimum_ll(burst_size, window_size, verbose=verbose, return_arrays=False)
        minimum_rho_dot, minimum_ll, rho_dots, lls = pbh.get_minimum_ll(burst_size, window_size, rho_dots=rho_dots,
                                                                        verbose=verbose)
        plt.plot(rho_dots, lls - minimum_ll, color=colors[ii],
                 label="burst size " + str(burst_size) + ", " + str(window_size) + "-s window")
    # plt.axvline(x=minimum_rho_dot, color="b", ls="--",
    #            label=("minimum -2lnL = %.2f at rho_dot = %.1f " % (minimum_ll, minimum_rho_dot)))
    plt.axhline(y=6.63, color="r", ls='--')
    plt.xlabel(r"Rate density of PBH evaporation (pc$^{-3}$ yr$^{-1}$)")
    plt.ylabel(r"-2$\Delta$lnL")
    plt.legend(loc='best')
    if xlog:
        plt.xscale('log')
    if grid:
        plt.grid(b=True)
    filename = save_hist + "_AllEvts.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print("Done!")

    return pbh


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

    # spec sim:
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


def test2():
    pbh = Pbh()
    pbh.get_TreeWithAllGamma(runNum=55480, nlines=1000)
    print(pbh.photon_df.head())
    return pbh
