from matplotlib import pyplot as plt
import numpy as np

from burstcalc.tools import get_n_expected
from burstcalc.io import load_pickle

try:
    import ROOT

    ROOT.PyConfig.StartGuiThread = False
except:
    print("Can't import ROOT, no related functionality possible")


def plot_n_expected(pbh, t_window, rs=np.arange(1e-2, 30, 1e-2), ax=None, color='b', label=None):
    n_exps = np.zeros_like(rs)
    for i, r in enumerate(rs):
        n_exps[i] = get_n_expected(pbh, t_window, r)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('N_expected')
    ax.plot(rs, n_exps, color=color, label=label)
    return ax


def plot_n_expected3(pbh, t_window, rs=np.arange(1e-2, 30, 1e-2), ax=None, color='b', label=None):
    n_exps = np.zeros_like(rs)
    for i, r in enumerate(rs):
        n_exps[i] = get_n_expected(pbh, t_window, r)
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Volume (pc$^3$)')
        ax.set_ylabel('N_expected')
    ax.plot(4. / 3. * np.pi * rs ** 3, n_exps, color=color, label=label)
    return ax


def plot_n_expected_all(pbh, t_windows, rs=np.arange(1e-2, 30, 1e-2), ax=None, colors=None, labels=None, show=True,
                        save=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Distance (pc)')
        ax.set_ylabel('N_expected')
    for i, t in enumerate(t_windows):
        if labels is not None:
            l = labels[i]
        else:
            l = None
        if colors is not None:
            plot_n_expected(pbh, t, rs=rs, ax=ax, color=colors[i], label=l)
        else:
            plot_n_expected(pbh, t, rs=rs, ax=ax, label=l)
    plt.legend(loc='best')
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()


def plot_residual_UL_n_expected(pbhs, rho_dots, ULs, colors=None, draw_grid=True, ylim=None,
                                filename="residual_UL_n_expected.png", show=True, ylog=False, verbose=True):
    n_expected = np.zeros(len(pbhs.burst_sizes_set))
    if not isinstance(rho_dots, list):
        rho_dots = [rho_dots]
    if colors is not None:
        if len(colors) != len(rho_dots):
            print("colors provided has a different length from rho_dots!")
            colors = None
    residual_dict = pbhs.get_residual_hist()
    sig_err = np.sqrt(np.array(pbhs.sig_burst_hist.values()).astype('float64'))
    bkg_err = np.sqrt(np.array(pbhs.avg_bkg_hist.values()).astype('float64'))
    res_err = np.sqrt(sig_err ** 2 + bkg_err ** 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs', capthick=0,
                label="Residual")
    for k, rho_dot in enumerate(rho_dots):
        for i, b in enumerate(pbhs.burst_sizes_set):
            n_expected[i] = pbhs.n_excess(rho_dot, pbhs.effective_volumes[b])
        if colors is not None:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], color=colors[k],
                    label=r"Expected excess of bursts $\dot{\rho}$=" + str(rho_dot))
        else:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:],
                    label=r"Expected excess of bursts $\dot{\rho}$=" + str(rho_dot))
        if verbose:
            print("Expected rate for rho_dot=%.1f is" % rho_dot)
            print(n_expected)

    ax.plot(residual_dict.keys()[1:], ULs, color='r', marker='v', label="90% UL Helene")
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel("Burst size")
    ax.set_ylabel("Counts")
    if ylim is not None:
        plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
    plt.legend(loc='best')
    if draw_grid:
        plt.grid(b=True)
    # plt.yscale('log')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    # return n_expected


def plot_Veff(pbh, window_sizes=[1, 10, 100], burst_sizes=range(2, 11), lss=['-', '--', ':'], cs=['r', 'b', 'k'],
              draw_grid=True, filename="Effective_volume.png"):
    for i, window_ in enumerate(window_sizes):
        Veffs = []
        for b in burst_sizes:
            Veffs.append(pbh.V_eff(b, window_))
        plt.plot(burst_sizes, Veffs, color=cs[i], ls=lss[i], label=("search window %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    plt.yscale('log')
    plt.xlabel("burst size")
    plt.ylabel(r"effective volume (pc$^3$)")
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_Veff_step(pbh, window_sizes=[1, 10, 100], burst_sizes=range(2, 11), lss=['-', ':', '--'], cs=['r', 'b', 'k'],
                   draw_grid=True, filename="Effective_volume.png"):
    for i, window_ in enumerate(window_sizes):
        Veffs = []
        for b in burst_sizes:
            Veffs.append(pbh.V_eff(b, window_))
        plt.step(burst_sizes, Veffs, color=cs[i], where='mid', linestyle=lss[i], label=(r"$\Delta$t = %d s" % window_))
    if draw_grid:
        plt.grid(b=True)
    plt.yscale('log')
    plt.xlabel("Burst size")
    plt.ylabel(r"Effective volume (pc$^3$)")
    plt.legend(loc='best')
    plt.ylim(1e-4, 1.1)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_residual_vs_n_expected(pbhs, rho_dots, colors=None, draw_grid=True, ylim=None,
                                filename="residual_vs_n_expected.png", show=True, ylog=False):
    n_expected = np.zeros(len(pbhs.burst_sizes_set))
    if not isinstance(rho_dots, list):
        rho_dots = [rho_dots]
    if colors is not None:
        if len(colors) != len(rho_dots):
            print("colors provided has a different length from rho_dots!")
            colors = None
    residual_dict = pbhs.get_residual_hist()
    sig_err = np.sqrt(np.array(pbhs.sig_burst_hist.values()).astype('float64'))
    bkg_err = np.sqrt(np.array(pbhs.avg_bkg_hist.values()).astype('float64'))
    res_err = np.sqrt(sig_err ** 2 + bkg_err ** 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(residual_dict.keys()[1:], residual_dict.values()[1:], xerr=0.5, yerr=res_err[1:], fmt='bs', capthick=0,
                label="Residual")
    for k, rho_dot in enumerate(rho_dots):
        for i, b in enumerate(pbhs.burst_sizes_set):
            n_expected[i] = pbhs.n_excess(rho_dot, pbhs.effective_volumes[b])
        if colors is not None:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:], color=colors[k],
                    label=r"Expected number of bursts $\dot{\rho}$=" + str(rho_dot))
        else:
            ax.plot(list(pbhs.burst_sizes_set)[1:], n_expected[1:],
                    label=r"Expected number of bursts $\dot{\rho}$=" + str(rho_dot))
    ax.axhline(y=0, color='gray', ls='--')
    ax.set_xlabel("Burst size")
    ax.set_ylabel("Counts")
    if ylim is not None:
        plt.ylim(ylim)
    if ylog:
        plt.yscale('log')
    plt.legend(loc='best')
    if draw_grid:
        plt.grid(b=True)
    # plt.yscale('log')
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()


def plot_pbh_ll_vs_rho_dots(pbhs_list, rho_dots=np.arange(0., 3.e5, 100), burst_size_thresh=2,
                            filename="ll_vs_rho_dots.png",
                            label_names=None, xlog=True, grid=True, plot_hline=True, show=False, xlim=None, ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if label_names is not None:
        assert len(pbhs_list) == len(label_names), "Check the length of label_names, doesn't match pbhs_list"
    for i, p in enumerate(pbhs_list):
        if label_names is not None:
            label_name = label_names[i] + " burst size " + str(burst_size_thresh) + ", " + str(
                p.window_size) + "-s window"
        else:
            label_name = "burst size " + str(burst_size_thresh) + ", " + str(p.window_size) + "-s window"
        minimum_rho_dot, minimum_ll, rho_dots, lls = p.get_minimum_ll(burst_size_thresh, p.window_size,
                                                                      rho_dots=rho_dots, verbose=p.verbose)
        plt.plot(rho_dots, lls - minimum_ll, label=label_name)
    # plt.axvline(x=minimum_rho_dot, color="b", ls="--",
    #            label=("minimum -2lnL = %.2f at rho_dot = %.1f " % (minimum_ll, minimum_rho_dot)))
    if plot_hline:
        plt.axhline(y=6.63, color="r", ls='--')
    plt.xlabel(r"Rate density of PBH evaporation (pc$^{-3}$ yr$^{-1}$)")
    plt.ylabel(r"-2$\Delta$lnL")
    plt.legend(loc='best')
    if xlog:
        plt.xscale('log')
    if grid:
        plt.grid(b=True)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    print("Done!")


def compare_run(runid, window_size=10, outfile=None, ylog=True, pkldir="batch_all_scramble_all_events", show=True):
    # compare burst histograms with Simon's
    if window_size == 1:
        test_pbh = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window1.0-s.pkl")
        tp = test_pbh.pbhs[0]
    elif window_size == 2:
        test_pbh2 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window2.0-s.pkl")
        tp = test_pbh2.pbhs[0]
    elif window_size == 5:
        test_pbh5 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window5.0-s.pkl")
        tp = test_pbh5.pbhs[0]
    elif window_size == 10:
        test_pbh10 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window10.0-s.pkl")
        tp = test_pbh10.pbhs[0]

    fname = "/raid/reedbuck/archs/PBHAnalysis_cori/MoreFinalResults/run_" + str(runid) + "_burst.root"

    rf = ROOT.TFile(fname, "read")
    if window_size == 1:
        h10b = rf.Get("1.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("1.0-second_run_" + str(runid) + "_Data");
    elif window_size == 2:
        h10b = rf.Get("2.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("2.0-second_run_" + str(runid) + "_Data");
    elif window_size == 5:
        h10b = rf.Get("5.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("5.0-second_run_" + str(runid) + "_Data");
    elif window_size == 10:
        h10b = rf.Get("10.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("10.0-second_run_" + str(runid) + "_Data");

    x = np.zeros(6)
    y = np.zeros(6)
    dy = np.zeros(6)

    xB = np.zeros(6)
    yB = np.zeros(6)
    dyB = np.zeros(6)

    for i in range(6):
        xB[i] = h10b.GetBinCenter(i + 1)
        yB[i] = h10b.GetBinContent(i + 1)
        dyB[i] = h10b.GetBinError(i + 1)
        x[i] = h10.GetBinCenter(i + 1)
        y[i] = h10.GetBinContent(i + 1)
        dy[i] = h10.GetBinError(i + 1)

    plt.errorbar(x, y, xerr=0.5, yerr=dy, label="Simon's signal",
                 fmt='v', color='r', ecolor='r', capthick=0)
    plt.errorbar(xB, yB, xerr=0.5, yerr=dyB, label="Simon's background",
                 fmt='<', color='b', ecolor='b', capthick=0)
    plt.errorbar(tp.sig_burst_hist.keys(), tp.sig_burst_hist.values(),
                 xerr=0.5, yerr=np.sqrt(np.array(tp.sig_burst_hist.values())), label="Qi's signal",
                 fmt='^', color='c', ecolor='c', capthick=0)
    plt.errorbar(tp.avg_bkg_hist.keys(), tp.avg_bkg_hist.values(),
                 xerr=0.5, yerr=np.sqrt(np.array(tp.avg_bkg_hist.values())), label="Qi's background",
                 fmt='>', color='m', ecolor='m', capthick=0)
    plt.title("run " + str(runid) + " " + str(window_size) + "-s window")
    plt.xlabel("Burst size")
    plt.ylabel("Counts")
    if ylog:
        plt.yscale('log')
    plt.legend()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    if show:
        plt.show()
    else:
        plt.clf()


def compare_run_ratio(runid, window_size=10, outfile=None, ylog=True, pkldir="batch_all_scramble_all_events",
                      show=True):
    if window_size == 1:
        test_pbh = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window1.0-s.pkl")
        tp = test_pbh.pbhs[0]
    elif window_size == 2:
        test_pbh2 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window2.0-s.pkl")
        tp = test_pbh2.pbhs[0]
    elif window_size == 5:
        test_pbh5 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window5.0-s.pkl")
        tp = test_pbh5.pbhs[0]
    elif window_size == 10:
        test_pbh10 = load_pickle(pkldir + "/pbhs_bkg_method_scramble_run" + str(runid) + "_window10.0-s.pkl")
        tp = test_pbh10.pbhs[0]

    fname = "/raid/reedbuck/archs/PBHAnalysis_cori/MoreFinalResults/run_" + str(runid) + "_burst.root"

    rf = ROOT.TFile(fname, "read")
    if window_size == 1:
        h10b = rf.Get("1.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("1.0-second_run_" + str(runid) + "_Data");
    elif window_size == 2:
        h10b = rf.Get("2.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("2.0-second_run_" + str(runid) + "_Data");
    elif window_size == 5:
        h10b = rf.Get("5.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("5.0-second_run_" + str(runid) + "_Data");
    elif window_size == 10:
        h10b = rf.Get("10.0-second_run_" + str(runid) + "_Background");
        h10 = rf.Get("10.0-second_run_" + str(runid) + "_Data");

    x = np.zeros(6)
    y = np.zeros(6)
    dy = np.zeros(6)

    xB = np.zeros(6)
    yB = np.zeros(6)
    dyB = np.zeros(6)

    for i in range(6):
        xB[i] = h10b.GetBinCenter(i + 1)
        yB[i] = h10b.GetBinContent(i + 1)
        dyB[i] = h10b.GetBinError(i + 1)
        x[i] = h10.GetBinCenter(i + 1)
        y[i] = h10.GetBinContent(i + 1)
        dy[i] = h10.GetBinError(i + 1)

    plt.errorbar(x, y * 1.0 / tp.sig_burst_hist.values(), xerr=0.5,
                 yerr=np.sqrt(dy ** 2 + np.array(tp.sig_burst_hist.values())), label="Ratio of signals",
                 fmt='v', color='r', ecolor='r', capthick=0)
    plt.errorbar(xB, yB * 1.0 / tp.avg_bkg_hist.values(), xerr=0.5,
                 yerr=np.sqrt(dyB ** 2 + np.array(tp.avg_bkg_hist.values())), label="Ratio of background",
                 fmt='<', color='b', ecolor='b', capthick=0)
    plt.title("run " + str(runid) + " " + str(window_size) + "-s window")
    plt.xlabel("Burst size")
    plt.ylabel("Simon's Counts / Qi's Counts")
    if ylog:
        plt.yscale('log')
    plt.legend()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    if show:
        plt.show()
    else:
        plt.clf()
