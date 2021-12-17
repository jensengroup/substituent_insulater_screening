import time
import os
import sys
import textwrap

import pandas as pd
import numpy as np

#sys.path.append("/groups/kemi/koerstz/opt/QMC/QMC")
sys.path.append("/home/koerstz/projects/gemma_part2/QMC_6.2") # for sunray
from qmconf import QMConf


def qsub_prep(batchname, script_path, cpus, mem):
    """ """
    pwd = os.getcwd()

    qsub_file = '''\
    #!/bin/sh
    #SBATCH --job-name={0}
    #SBATCH --cpus-per-task={1}
    #SBATCH --mem={2}
    #SBATCH --ntasks=1
    #SBATCH --error={3}/{0}.stderr
    #SBATCH --output={3}/{0}.stdout
    #SBATCH --time=24:00:00
    #SBATCH --partition=coms
    #SBATCH --no-requeue

    cd /scratch/$SLURM_JOB_ID

    # copy batch file
    cp {3}/{0}.csv .

    # run python code
    /home/koerstz/anaconda3/envs/quantum_ml/bin/python {4} {0}.csv {1}

    # copy data back
    cp *pkl {3}

    '''.format(batchname, cpus, mem, pwd, script_path)

    with open(batchname + "_qsub.tmp", 'w') as qsub:
        qsub.write(textwrap.dedent(qsub_file))

    return batchname + "_qsub.tmp"


def submit_job(csv, script, mem, cpus, nodes):

    qsub_name = qsub_prep(csv, script, cpus, mem)

    batch_id = os.popen("sbatch " + qsub_name).read()
    batch_id = batch_id.strip().split()[-1]

    return int(batch_id)


def run_calculations(csv_names, script, mem, cpus, nodes):
    ''' '''

    submitted_jobs = set()
    for csv in csv_names:
        batch_id = submit_job(csv, script, mem, cpus, nodes)
        submitted_jobs.add(batch_id)

        if len(submitted_jobs) >= nodes:

            while True:
                output = os.popen("squeue -u koerstz").readlines()[1:]
                all_running_jobs = set([int(job.split()[0]) for job in output])

                if len(all_running_jobs & submitted_jobs) >= nodes: # intersect
                    time.sleep(10)
                else:
                    # remove finished jobs
                    finished_jobs = submitted_jobs - all_running_jobs
                    for job in finished_jobs:
                        submitted_jobs.remove(job)
                    break


if __name__ == "__main__":

    # input params
    cpus = 2
    mem = "10GB"

    nodes = 250
    chunk_size = 1

    #script = '/home/koerstz/projects/gemma_part2/gemma_conf_search.py'
    script = '/home/koerstz/projects/gemma_part2/gemma_v2.py'

    data_file = sys.argv[1]

    ##########################################################################
    #
    ##########################################################################

    # import data
    data = pd.read_csv(data_file)

    # split data into chunks
    chunked_data = [data[i:i+chunk_size] for i in range(0, data.shape[0], chunk_size)]

    chunk_names = list()
    for idx, chunk in enumerate(chunked_data):
        chunk_name = "smiles_batch-{}".format(idx)
        chunk.to_csv(chunk_name + ".csv", index=False)

        chunk_names.append(chunk_name)

    # run calculations on nodes
    run_calculations(chunk_names, script, mem, cpus, nodes)
