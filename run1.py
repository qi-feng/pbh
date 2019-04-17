#!/usr/local/bin/python3
import click
# import logging

from burstcalc.io import load_pickle, dump_pickle
from burstcalc.burst import BurstFile

# logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--debug', '-d', is_flag=True)
@click.option('--run_number', '-r', nargs=1, type=int, help="Run Number")
@click.option('--infile_path', '-p', nargs=1, type=click.Path(exists=True), default="TestFiles/",
              help="Path to directory containing input files [TestFiles/]")
@click.option('--cuts', '-c', nargs=1, type=str, default="soft",
              help="gamma/hadron cuts [soft]")
@click.option('--using_ed', '-ed', is_flag=True,
              help="Set if reading in ED files, if not set assumes VEGAS.")
@click.option('--num_ev_to_read', '-n', nargs=1, type=int, default=None,
              help="Number of events to read, if None (default), all events in root file")
@click.option('--window_size', '-w', nargs=1, type=float, default=1,
              help="Window size in s [1]")
@click.option('--num_scramble', '-s', nargs=1, type=int, default=5,
              help="Number of scrambles [5]")
@click.option('--bg_method', '-b', nargs=1, type=str, default="scramble_times",
              help="Background method, can be either scramble_times or rando (details??) [scramble_times]")
@click.option('--random_method', '-m', nargs=1, type=str, default="avg",
              help="Random method [scramble_times]")
@click.option('--load_from_pickle', '-l', is_flag=True, help="Load data from pickle rather than root file")
def run1(run_number, infile_path, using_ed, cuts, debug, num_ev_to_read, window_size, num_scramble, bg_method,
         random_method, load_from_pickle):
    pickle_path = "{0:s}/{1:d}.pickle".format(infile_path, run_number)

    if load_from_pickle:
        # logger.info("Loading from pickle file {0:s}".format(pickle_path))
        burst = load_pickle(pickle_path)

        #note - we need to reinitialize the logger when reloading from file
        burst.initiate_logger(debug)
    else:
        burst = BurstFile(run_number, infile_path, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                           debug=debug)
        # load the cuts
        # TODO: this needs editing to load everything from file
        burst.set_cuts(cuts=cuts)
        # load the root tree and the psf
        burst.get_tree_with_all_gamma()

        # plot the histogram of the psf for debugging purposes
        # burst1.plot_psf_2dhist()

        # caluclate the number of bursts (this is the signal search)
        burst.signal_burst_search(window_size=window_size)

        # calculate the background burst distribution
        burst.estimate_bkg_burst(window_size=window_size,
                                  rando_method=random_method,
                                  method=bg_method,
                                  copy=True,
                                  n_scramble=num_scramble)

        dump_pickle(burst, pickle_path)

    burst.logger.info("Burst Hist")
    burst.logger.info(burst.signal_burst_hist)

    burst.logger.info("Background Hist")
    burst.logger.info(burst.avg_bkg_hist)

    # burst.calculate_stats()
    #
    # burst.plot_burst_hist()


if __name__ == '__main__':
    run1()
