#!/usr/local/bin/python3
import click

from burstcalc.burst import BurstFile

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--debug', '-d', is_flag=True)
@click.option('--run_number', '-r', nargs=1, type=int, help="Run Number")
@click.option('--infile_path', '-p', nargs = 1, type=click.Path(exists=True), help="Path to directory containing input files [TestFiles/]",
              default="TestFiles/")
@click.option('--cuts', '-c', nargs=1, type=str, help="gamma/hadron cuts [soft]", default="soft")
@click.option('--using_ed', '-ed', is_flag=True)
@click.option('--num_ev_to_read', '-n', nargs=1, type=int, default=None,
              help="Number of events to read, if None (default), all events in root file")
@click.option('--window_size', '-w', nargs=1, help="Window size in s", default = 1)
def run1(run_number, infile_path, using_ed, cuts, debug, num_ev_to_read, window_size):
    burst1 = BurstFile(run_number, infile_path, using_ed=using_ed, num_ev_to_read=num_ev_to_read,
                       debug=debug)
    # load the cuts (TODO: this needs editing to load everything from file)
    burst1.set_cuts(cuts=cuts)
    # load the root tree and the psf
    # burst1.get_tree_with_all_gamma()
    burst1.plot_psf_2dhist()
#    _sig_burst_hist, _sig_burst_dict = burst1.sig_burst_search(window_size=window_size)
# _avg_bkg_hist, _bkg_burst_dicts = pbh_.estimate_bkg_burst(window_size=self.window_size,
#                                                           rando_method=self.rando_method,
#                                                           method=self.bkg_method, copy=True,
#                                                           n_scramble=self.N_scramble,
#                                                           return_burst_dict=True, verbose=self.verbose)
# pbh_.get_run_summary()


if __name__ == '__main__':
    run1()
