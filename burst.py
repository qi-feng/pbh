import os
import pandas as pd
import socket
from optparse import OptionParser

from burst.process import process_one_run


def qsub_job_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                     script_dir='/raid/reedbuck/qfeng/pbh/', overwrite=True, hostname=None, walltime=48):
    if window_size < 1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f' % (filename, window_size))
    # data_base_dir = '/raid/reedbuck/veritas/data/'
    # script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["run_number"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "burst/pbh.py"
            if window_size < 1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.pbs' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.pbs' % (run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_" + str(bkg_method) + "_run" + str(run_num) + "_window" + str(
                window_size) + "-s.pkl"
            # if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                # print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***' % pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    # sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log' % (run_num, window_size)
            # script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#PBS -e %s\n' % os.path.join(script_dir, 'qsub_%s.err' % scriptname))
                script.write('#PBS -o %s\n' % os.path.join(script_dir, 'qsub_%s.log' % scriptname))
                script.write('#PBS -l walltime=' + str(walltime) + ':00:00\n')
                script.write('#PBS -l pvmem=5gb\n')
                script.write('cd %s\n' % script_dir)
                if plot:
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
            script.close()
            # isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            isend_command = 'qsub -l nodes=%s -q batch -V %s' % (hostname, scriptfullname)

            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise


def qsub_cori_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                      script_dir='/global/cscratch1/sd/qifeng/pbh/', overwrite=True, hostname=None, walltime=48):
    if window_size < 1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f' % (filename, window_size))
    # data_base_dir = '/raid/reedbuck/veritas/data/'
    # script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["run_number"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "pbh.py"
            if window_size < 1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.pbs' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.pbs' % (run_num, window_size)
            # scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_" + str(bkg_method) + "_run" + str(run_num) + "_window" + str(
                window_size) + "-s.pkl"
            # if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                # print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***' % pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    # sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log' % (run_num, window_size)
            # logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            # script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#!/bin/bash -l \n\n')
                script.write('#SBATCH -p shared \n')
                script.write('#SBATCH -N 1\n')
                script.write('#SBATCH -L SCRATCH\n')
                script.write('#SBATCH -e %s\n' % os.path.join(script_dir, 'qsub_%s.err' % scriptname))
                script.write('#SBATCH -o %s\n' % os.path.join(script_dir, 'qsub_%s.log' % scriptname))
                script.write('#SBATCH -t %s\n' % (str(walltime) + ':00:00\n'))
                # script.write('#SBATCH -l pvmem=5gb\n')
                script.write("source /global/homes/q/qifeng/.bashrc.ext")
                script.write('cd %s\n' % script_dir)
                if plot:
                    # script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    # script.write('python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    # script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
            script.close()
            # isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            # isend_command = 'qsub -l nodes=%s -q batch -V %s'%(hostname, scriptfullname)
            isend_command = 'sbatch -C haswell %s' % (scriptfullname)

            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise


def qsub_tehanu_runlist(filename="pbh_runlist.txt", window_size=10, plot=False, bkg_method="scramble",
                        script_dir='/a/data/tehanu/qifeng/pbh', overwrite=True, hostname=None, walltime=48):
    if window_size < 1:
        print('Submitting jobs for runlist %s with search window size %.1g' % (filename, window_size))
    else:
        print('Submitting jobs for runlist %s with search window size %.1f' % (filename, window_size))
    # data_base_dir = '/raid/reedbuck/veritas/data/'
    # script_dir = '/raid/reedbuck/qfeng/pbh/'

    runlist_df = pd.read_csv(filename, header=None)
    runlist_df.columns = ["run_number"]
    runlist = runlist_df.runNum.values
    if hostname is None:
        hostname = socket.gethostname()
    for run_num in runlist:
        try:
            pyscriptname = "pbh.py"
            if window_size < 1:
                scriptname = 'pbhs_run%d_window_size%.4f-s.sh' % (run_num, window_size)
                condor_scriptname = 'pbhs_run%d_window_size%.4f-s.condor' % (run_num, window_size)
            else:
                scriptname = 'pbhs_run%d_window_size%d-s.sh' % (run_num, window_size)
                condor_scriptname = 'pbhs_run%d_window_size%d-s.condor' % (run_num, window_size)
            # scriptname = 'pbhs_run%d_window_size%d-s.pbs'%(run_num, window_size)
            scriptfullname = os.path.join(script_dir, scriptname)
            condor_scriptname = os.path.join(script_dir, condor_scriptname)
            pyscriptname = os.path.join(script_dir, pyscriptname)
            pklname = "pbhs_bkg_method_" + str(bkg_method) + "_run" + str(run_num) + "_window" + str(
                window_size) + "-s.pkl"
            # if os.path.exists(scriptfullname):
            if os.path.exists(pklname):
                # print('*** Warning: script already exists: %s ***'%scriptfullname)
                print('*** Warning: pickle file already exists: %s ***' % pklname)
                if not overwrite:
                    print("Aborting")
                    continue
                    # sys.exit(1)
                print('Overwriting...')
            if window_size < 1:
                logfilename = 'pbhs_run%d_window_size%.4f-s.log' % (run_num, window_size)
            else:
                logfilename = 'pbhs_run%d_window_size%d-s.log' % (run_num, window_size)
            # logfilename = 'pbhs_run%d_window_size%d-s.log'%(run_num, window_size)
            # script = open(scriptfullname, 'w')
            with open(scriptfullname, 'w') as script:
                script.write('#!/bin/bash  \n\n')
                script.write('date \n')
                script.write('hostname \n')
                script.write('cd {} \n'.format(script_dir))
                script.write('pwd \n')
                script.write('shopt -s expand_aliases \n')
                script.write('source /usr/nevis/adm/nevis-init.sh \n')
                script.write('source /a/home/tehanu/qifeng/.bashrc \n')

                if plot:
                    # script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    # script.write('python %s -r %d -w %d -b %s -p >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s -p >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                else:
                    # script.write('srun -n 1 -C haswell python %s -r %d -w %d -b %s >> %s\n'%(pyscriptname, run_num, window_size, bkg_method, logfilename))
                    if window_size < 1:
                        script.write('python %s -r %d -w %.4f -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))
                    else:
                        script.write('python %s -r %d -w %d -b %s >> %s\n' % (
                            pyscriptname, run_num, window_size, bkg_method, logfilename))

                script.write('pwd \n')
                script.write('whoami \n')
                script.write('date \n')

            with open(condor_scriptname, 'w') as condor_script:
                condor_script.write('Universe  = vanilla \n')
                condor_script.write('Executable = {} \n'.format(scriptfullname))
                condor_script.write('Log = {}/condor_{}.log \n'.format('/'.join(scriptfullname.split('/')[:-2]),
                                                                       '.'.join(logfilename.split('.')[:-1])))
                condor_script.write('Error =  {}/condor_{}.err \n'.format('/'.join(scriptfullname.split('/')[:-2]),
                                                                          '.'.join(logfilename.split('.')[:-1])))
                condor_script.write('Output =  {}/condor_{}.out \n'.format('/'.join(scriptfullname.split('/')[:-2]),
                                                                           '.'.join(logfilename.split('.')[:-1])))
                condor_script.write('should_transfer_files = YES \n')
                condor_script.write('WhenToTransferOutput = ON_EXIT \n')
                # condor_script.write('Requirements = (machine == \"tehanu.nevis.columbia.edu\" || machine== \"ged.nevis.columbia.edu\" || machine== \"serret.nevis.columbia.edu\") \n')
                condor_script.write(
                    'Requirements = (machine== \"ged.nevis.columbia.edu\" || machine== \"serret.nevis.columbia.edu\") \n')
                condor_script.write('Notification = NEVER \n')
                condor_script.write('getenv = True \n')
                condor_script.write('Queue \n')

            # isend_command = 'qsub -l nodes=reedbuck -q batch -V %s'%scriptfullname
            # isend_command = 'qsub -l nodes=%s -q batch -V %s'%(hostname, scriptfullname)
            # isend_command = 'sbatch -C haswell %s'%(scriptfullname)
            isend_command = 'condor_submit {}'.format(condor_scriptname)

            print(isend_command)
            os.system(isend_command)
            print("Run %d sent to queue." % run_num)
        except:
            print("*** Can't process run: %d ***" % run_num)
            raise


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-l", "--list", dest="runlist", default=None)
    parser.add_option("-r", "--run", dest="run", type="int", default=None)
    parser.add_option("-w", "--window", dest="window", type="float", default=10)
    # parser.add_option("-p","--plot",dest="plot",default=False)
    parser.add_option("-p", "--plot", action="store_true", dest="plot", default=False)
    parser.add_option("-b", "--bkg_method", dest="bkg_method", default="scramble")
    parser.add_option("-m", "--makeup", action="store_false", dest="overwrite", default=True)
    parser.add_option("-t", "--walltime", dest="walltime", type="int", default=48)
    parser.add_option("-c", "--cori", action="store_true", dest="cori", default=False)
    parser.add_option("--tehanu", action="store_true", dest="tehanu", default=False)
    # parser.add_option("--rho_dots",dest="rho_dots", default=np.arange(0, 2e7, 1e4))
    # parser.add_option("-inner","--innerHi",dest="innerHi",default=True)
    (options, args) = parser.parse_args()

    if options.runlist is not None:
        # print('Submitting jobs for runlist %s with search window size %.1f'%(options.runlist, options.window))
        if options.cori:
            qsub_cori_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                              bkg_method=options.bkg_method, script_dir=os.getcwd(), overwrite=options.overwrite,
                              walltime=options.walltime)
        elif options.tehanu:
            qsub_tehanu_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                                bkg_method=options.bkg_method, overwrite=options.overwrite, walltime=options.walltime)
        else:
            qsub_job_runlist(filename=options.runlist, window_size=options.window, plot=options.plot,
                             bkg_method=options.bkg_method, script_dir=os.getcwd(), overwrite=options.overwrite,
                             walltime=options.walltime)

    if options.run is not None:
        print('\n\n#########################################')
        if options.window < 1:
            print('Processing run %d with search window size %.1g' % (options.run, options.window))
        else:
            print('Processing run %d with search window size %.1f' % (options.run, options.window))
        process_one_run(options.run, options.window, bkg_method=options.bkg_method, plot=options.plot)
