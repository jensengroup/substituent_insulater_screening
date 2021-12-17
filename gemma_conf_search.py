import sys
import numpy as np
import pandas as pd

from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append("/home/koerstz/projects/gemma_part2/QMC_6.2")
from qmmol import QMMol
from qmconf import QMConf
from calculator.xtb import xTB

from conformers.create_conformers import RotatableBonds


def gs_conformer_search(name, rdkit_conf, chrg, mult, cpus):
    """ ground state conformer search """
    
    charged = False # hard coded for mogens

    # create conformers
    qmmol = QMMol()
    qmmol.add_conformer(rdkit_conf, fmt='rdkit', label=name, 
                        charged_fragments=charged, set_initial=True)
    
    #print(qmmol.conformers)
    #print(qmmol.conformers[0].write_xyz())

    #quit()


    num_confs = 5 
    qmmol.create_random_conformers(threads=cpus, num_confs=num_confs) 
    
    print(len(qmmol.conformers))    
    for conf in qmmol.conformers:
        print(conf.label)
        conf.write_xyz()
    
    quit()

    xtb_params = {'method': 'gfn2',
                  'opt': 'opt',
                  'cpus': 1}

    qmmol.calc = xTB(parameters=xtb_params)
    qmmol.optimize(num_procs=cpus, keep_files=True)
    
    #for conf in qmmol.conformers:
    #    print(conf.label, conf.results['energy'])
    #    conf.write_xyz()

    # Get most stable conformer. If most stable conformer
    # not identical to initial conf try second lowest.
    initial_smi = Chem.MolToSmiles(Chem.RemoveHs(qmmol.initial_conformer.get_rdkit_mol()))
    
    low_energy_conf = qmmol.nlowest(1)[0]
    try:
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(low_energy_conf.get_rdkit_mol()))
    except:
        conf_smi = 'fail'

    i = 1
    while initial_smi != conf_smi:
        low_energy_conf = qmmol.nlowest(i+1)[-1]
        try:
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(low_energy_conf.get_rdkit_mol()))
        except:
            conf_smi = 'fail'
        
        i += 1
        
        if len(qmmol.conformers) < i:
            sys.exit('no conformers match the initial input')

    return low_energy_conf


def gs_gemma(tup): #name, smi, chrg, mult, cps):
    """GS conformers search given a smiles string  """
    
    cps = 1
    name, smi, chrg, mult = tup.comp_name, tup.smiles, tup.charge, tup.multiplicity

    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    
    AllChem.EmbedMolecule(mol)
    mol = Chem.AddHs(mol)
    Chem.MolToMolFile(mol, name + '.sdf')

    rdkit_conf = mol.GetConformer()

    qmconf = gs_conformer_search(name, rdkit_conf, chrg, mult, cps)
    #print(qmconf.results, qmconf.label)

    return qmconf



if __name__ == '__main__':

    cpus = 2

    data = pd.read_csv(sys.argv[1])
    # find storage energy
    
    compound_list = list()
    for compound in data.itertuples():
        mol =  gs_gemma(compound)
        
        compound_list.append({'comp_name': compound.comp_name,
                              'mol': mol})
    
    data_out = pd.DataFrame(compound_list)
    data_out.to_pickle(sys.argv[1].split('.')[0] + '.pkl')
