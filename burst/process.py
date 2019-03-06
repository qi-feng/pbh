import numpy as np

from burst.io import dump_pickle
from burst.pbh import Pbh_combined


def process_one_run(run, window_size, rho_dots=np.arange(0., 3.e5, 100), plot=False, bkg_method="scramble"):
    pbhs = Pbh_combined(window_size)
    pbhs.rho_dots = rho_dots
    pbhs.bkg_method = bkg_method
    try:
        pbhs.add_run(run)
        print("Run %d processed." % run)
    except:
        print("*** Bad run: %d ***" % run)
        raise
    dump_pickle(pbhs,
                "pbhs_bkg_method_" + str(bkg_method) + "_run" + str(run) + "_window" + str(window_size) + "-s.pkl")
    if plot:
        pbhs.plot_ll_vs_rho_dots(save_hist="ll_vs_rho_dots_run" + str(run) + "_window" + str(window_size) + "-s")
        pbhs.plot_burst_hist(filename="burst_hists_run" + str(run) + "_window" + str(window_size) + "-s.png",
                             title="Burst histogram run" + str(run) + "" + str(window_size) + "-s window ",
                             plt_log=True, error="Poisson")
