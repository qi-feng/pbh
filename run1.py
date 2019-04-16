#!/usr/local/bin/python3
import click

from burstcalc.burst import BurstFile

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
@click.option('--window_size', '-w', nargs=1, type=int, default=1,
              help="Window size in s [1]")
@click.option('--num_scramble', '-s', nargs=1, type=int, default=10,
              help="Number of scrambles [10]")
@click.option('--bg_method', '-b', nargs=1, type=str, default="scramble_times",
              help="Background method, can be either scramble_times or rando (details??) [scramble_times]")
@click.option('--random_method', '-m', nargs=1, type=str, default="avg",
              help="Random method [scramble_times]")
def run1(run_number, infile_path, using_ed, cuts, debug, num_ev_to_read, window_size, num_scramble, bg_method,
         random_method):
    burst1 = BurstFile(run_number, infile_path, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                       debug=debug)
    # load the cuts (TODO: this needs editing to load everything from file)
    burst1.set_cuts(cuts=cuts)
    # load the root tree and the psf
    burst1.get_tree_with_all_gamma()

    # plot the histogram of the psf for debugging purposes
    # burst1.plot_psf_2dhist()

    # caluclate the number of bursts (this is the signal search)
    burst1.sig_burst_search(window_size=window_size)

    # calculate the background burst distribution
    burst1.estimate_bkg_burst(window_size=window_size,
                              rando_method=random_method,
                              method=bg_method,
                              copy=True,
                              n_scramble=num_scramble)

    # burst1.get_run_summary()


if __name__ == '__main__':
    run1()
