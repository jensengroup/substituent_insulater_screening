import sys
import os
import numpy as np
import pandas as pd

from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append("/home/koerstz/projects/gemma_part2/QMC_6.2")
from qmmol import QMMol
from qmconf import QMConf
from calculator.xtb import xTB


if __name__ == '__main__':

    df = pd.read_pickle(sys.argv[1])
    df.iloc[0].best_conf.write_xyz()
#    dfs = []
#    for _file in os.listdir('.'):
#        if _file.endswith('.pkl'):
#            dfs.append(pd.read_pickle(_file))
#    
#    data = pd.concat(dfs, sort=True)
#    for row in data.itertuples():
#        print(row.comp_name)
#    #print(data)
