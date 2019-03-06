import sys

if (sys.version_info > (3, 0)):
    # Python 3 code in this block
    import _pickle as pickle
else:
    # Python 2 code in this block
    import cPickle as pickle

import pandas as pd
import tables

from burst.pbh import Pbh_combined


def load_hdf5(fname):
    with tables.open_file(fname) as h5f:
        data = h5f.root.image[:]
    return data


def save_hdf5(data, ofname, compress=False, complevel=5, complib='zlib'):
    with tables.open_file(ofname, 'w') as h5f:
        atom = tables.Atom.from_dtype(data.dtype)
        shape = data.shape
        if compress:
            filters = tables.Filters(complevel=complevel, complib=complib)
            ca = h5f.create_carray(h5f.root, 'image', atom, shape, filters=filters)
        else:
            ca = h5f.create_carray(h5f.root, 'image', atom, shape)
        ca[:] = data[:]


def load_pickle(f):
    inputfile = open(f, 'rb')
    loaded = pickle.load(inputfile)
    inputfile.close()
    return loaded


def dump_pickle(a, f):
    output = open(f, 'wb')
    pickle.dump(a, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()


def combine_pbhs_from_pickle_list(list_of_pbhs_pickle, outfile="pbhs_combined"):
    list_df = pd.read_csv(list_of_pbhs_pickle, header=None)
    list_df.columns = ["pickle_file"]
    list_of_pbhs = list_df.pickle_file.values
    first_pbh = load_pickle(list_of_pbhs[0])
    window_size = first_pbh.window_size
    pbhs_combined = Pbh_combined(window_size)
    pbhs_combined.rho_dots = first_pbh.rho_dots
    for pbhs_pkl_ in list_of_pbhs:
        pbhs_ = load_pickle(pbhs_pkl_)
        assert window_size == pbhs_.window_size, "The input list contains pbhs objects with different window sizes!"
        pbhs_combined.add_pbh(pbhs_)
    dump_pickle(pbhs_combined, outfile + "_window" + str(window_size) + "-s_" + str(list_of_pbhs.shape[0]) + "runs.pkl")
    return pbhs_combined


def combine_from_pickle_list(listname, window_size, filetag="", burst_size_threshold=2, rho_dots=None,
                             upper_burst_size=None):
    pbhs_combined_all_ = combine_pbhs_from_pickle_list(listname)
    print("Total exposure time is %.2f hrs" % (pbhs_combined_all_.total_time_year * 365.25 * 24))
    pbhs_combined_all_.get_ULs(burst_size_threshold=burst_size_threshold, rho_dots=rho_dots,
                               upper_burst_size=upper_burst_size)
    print("The effective volume above burst size 2 is %.6f pc^3" % (pbhs_combined_all_.effective_volumes[2]))
    print("There are %d runs in total" % (len(pbhs_combined_all_.runNums)))
    total_N_runs = len(pbhs_combined_all_.runNums)
    pbhs_combined_all_.plot_burst_hist(
        filename="burst_hists_test_" + str(filetag) + "_window" + str(window_size) + "-s_all" + str(
            total_N_runs) + "runs.png",
        title="Burst histogram " + str(window_size) + "-s window " + str(total_N_runs) + " runs", plt_log=True,
        error="Poisson")
    pbhs_combined_all_.plot_ll_vs_rho_dots(
        save_hist="ll_vs_rho_dots_test_" + str(filetag) + "_window" + str(window_size) + "-s_all" + str(
            total_N_runs) + "runs")
    return pbhs_combined_all_
