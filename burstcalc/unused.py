import shutil

import pandas as pd
import numpy as np

from burstcalc.pbh import Pbh
from burstcalc.io import combine_from_pickle_list


def filter_good_runlist(infile="batch_all_v3/runlist_Final.txt", outfile="goodruns.txt"):
    df_run = pd.read_csv(infile)
    df_run.columns = ['run']
    bad_runs = []
    no_ea = []
    for run in df_run.values.flatten():
        p_ = Pbh()
        p_.read_ed_file(run)
        all_gamma_tree_name = "run_" + str(run) + "/stereo/TreeWithAllGamma"
        all_gamma_tree = p_.root_file.Get(all_gamma_tree_name)
        es_ = []

        ea_Name = "run_" + str(run) + "/stereo/EffectiveAreas/gMeanEffectiveArea"
        ea = p_.root_file.Get(ea_Name)
        try:
            ea.GetN()
        except:
            print("Empty EAs for run {}".format(run))
            no_ea.append(run)
            continue

        for i, event in enumerate(all_gamma_tree):
            if i > 100:
                break
            es_.append(event.Energy)

        if np.mean(np.asarray(es_)) == -99:
            print("Energy not filled for run {}".format(run))
            bad_runs.append(run)

    good_run = df_run[~df_run.run.isin(bad_runs)]
    good_run = good_run[~good_run.run.isin(no_ea)]
    print("Saving {} good runs to file {}".format(good_run.shape[0], outfile))
    good_run.to_csv(outfile, header=None, index=False)
    return good_run


def jackknife_runlist(runlist="pbh_1-s_scramble_all_90eff_list.txt", num_samples=5, run=True, window_size=1,
                      filetag="scramble_all_90eff", upper_burst_size=None, start_subsample=0):
    rlist = []
    nline = 0
    with open(runlist, 'r') as f:
        for line in f:
            rlist.append(line)
            nline += 1

    # chunk_length = nline//num_samples*(num_samples-1)
    for i in range(start_subsample, num_samples):
        outfilename = 'file' + str(i) + 'out_of' + str(num_samples) + runlist
        outfile = open(outfilename, 'w')
        subsample = [x for k, x in enumerate(rlist) if k % num_samples != i]
        # outfile.write("".join(rlist[chunk_length*i:chunk_length*(i+1)]))
        outfile.write("".join(subsample))
        outfile.close()
        chunk_length = len(subsample)

        if run:
            pbhs_combined_cv = combine_from_pickle_list(outfilename, window_size,
                                                        filetag=filetag + '_' + str(i) + 'out_of' + str(num_samples),
                                                        upper_burst_size=upper_burst_size)
            shutil.copyfile(
                'pbhs_combined_window' + str("{:.1f}".format(window_size)) + '-s_' + str(chunk_length) + 'runs.pkl',
                'pbhs_combined_window' + str("{:.1f}".format(window_size)) + '-s_' + str(
                    chunk_length) + 'runs_sub' + str(i) + '.pkl')
            print("Final UL is {0}".format(pbhs_combined_cv.rho_dot_ULs))
