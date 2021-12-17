import sys
import time
import os
import shutil
import copy
from collections import namedtuple
import itertools
import pandas as pd

from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdMolTransforms, rdMolAlign

sys.path.append("/home/koerstz/projects/gemma_part2/QMC_6.2")
from qmmol import QMMol
from qmconf import QMConf
from clustering.butina_clustering import butina_clustering_m

from xyz2mol.xyz2mol import AC2mol

import openbabel

def sdf2xyz(name):
    
    cmd = f'babel {name}.sdf -oxyz {name}.xyz'
    os.system(cmd)
    os.remove(name + '.sdf')


def runxtb(name):
    
    # set xtb path
    os.environ['XTBHOME'] = '/opt/xtb/6.2'
    os.environ['XTBPATH'] = '/opt/xtb/6.2/bin'
    
    # set cpu info
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # run program
    os.makedirs(name)
    os.rename(name + '.xyz', f'{name}/{name}.xyz')
    
    os.chdir(name)
    cmd = f'/opt/xtb/6.2/bin/xtb {name}.xyz --gfn2 --opt crude'
    output = os.popen(cmd).read()

    # move opt xyz files back
    os.rename('xtbopt.xyz', f'../{name}.xyz')
    os.rename('xtbopt.log', f'../{name}_opt.log')
    
    os.chdir('..')
    shutil.rmtree(name)


def xyz2qmconf(name):
    
    # get SCF energy
    with open(name + '.xyz', 'r') as f:
        output = f.readlines()
        energy = float(output[1].split()[2])
    
    qmconf = QMConf(name + '.xyz', fmt='xyz', label=name,
                    charged_fragments=False)
    qmconf.results['energy'] = energy

    return qmconf 

def make_qmmol(qmconfs):

    qmmol = QMMol()

    qmmol.charge = 0
    qmmol.multiplicity = 1
    qmmol.charged_fragments = False
    
    for conf in qmconfs:
        qmmol.conformers.append(conf)

    return qmmol

def get_xtb_opt_conf(name):
    sdf2xyz(name)
    runxtb(name)
    
    return xyz2qmconf(name)

def find_dihedral_idx(mol,smarts_patt):
    patt_mol = Chem.MolFromSmarts(smarts_patt)
    matches = mol.GetSubstructMatches(patt_mol)

    unique_match = list()
    match_list = list()
    for match in matches:
        if match[:3] not in match_list:
            unique_match.append(match)
            match_list.append(match[:3])

    if len(unique_match) != 2:
        print("more than two dihedrals in " + filename)
        quit()
    
    return unique_match


def sample_contact(rdkit, theta):
    """ Sample the dihedral of the electrode contact
    for now just rotate one end.
    """
    
    Chem.SanitizeMol(rdkit)
    initconf = rdkit.GetConformer()
    
    # set outer most dihedral to 180 degrees.
    smarts_patt = "C-S-C-[C,Si,Ge;H0]"
    outer_dihedral_idx = find_dihedral_idx(rdkit, smarts_patt)
    for k, i, j, l in outer_dihedral_idx:
        rdMolTransforms.SetDihedralDeg(initconf, k,i,j,l, 180.0)


    # sample the dihedral
    patt = "S-C-[C,Si,Ge;H0]-[C,Si,Ge]"
    dihedral_idx = find_dihedral_idx(rdkit, patt)[:1] # remoce [:1] to rotate both ends

    new_angles = list()
    for k, i, j, l in dihedral_idx:
        init_dihedral_angle = rdMolTransforms.GetDihedralDeg(initconf, k,i,j,l)
        new_angles.append([init_dihedral_angle + x*theta for x in range(int(360./theta))])
    
    angle_combinations = list(itertools.product(*new_angles)) # all combinations.
    

    for dihedrals in angle_combinations:
        for (k,i,j,l), angle in zip(dihedral_idx, dihedrals):
            rdMolTransforms.SetDihedralDeg(initconf, k,i,j,l, angle )

        rdkit.AddConformer(initconf, assignId=True)
    
    rdMolAlign.AlignMolConformers(rdkit)

    return rdkit


def write_rdkit_confs(rdkit, name):
    """ Write xyz of rdkit mol """
    
    xyz = ''
    for idx, conf in enumerate(rdkit.GetConformers()):
        if idx == 0: # skip first confs - identical to second
            continue 
        
        sdf_txt = sdf_txt = Chem.SDWriter.GetText(rdkit, conf.GetId())
        m = Chem.MolFromMolBlock(sdf_txt, removeHs=False)
            
        conf_name =  name + "-" + str(idx-1)
        m.SetProp("_Name", conf_name)
        
        # Convert sdf to xyz
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "xyz")
        
        new_conf_fmt = openbabel.OBMol()
        obConversion.ReadString(new_conf_fmt, Chem.MolToMolBlock(m))
        
        with open(name + f'-{idx}.xyz', 'w') as f:
            f.write(obConversion.WriteString(new_conf_fmt))

        #xyz += obConversion.WriteString(new_conf_fmt)
    
    #with open(name+'.xyz',  'w') as f:
    #    f.write(xyz)


if __name__ == '__main__':

    numConfs = int(100)
    max_ff_iters = int(1000)
    num_cpus = int(46)

    
    csv = pd.read_csv(sys.argv[1])
    
    for row in csv.itertuples():
        mol_name = row.comp_name
        smi = row.smiles

        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        orgAC = rdmolops.GetAdjacencyMatrix(mol, useBO=True) # to create final rdkit mol
        
        # Do i wan't to use useRandomCoods=True?
        # Perhaps it should just be the fallback method if it doesn't work with False settings. 
        # It is a more stable procedure. 
        t = time.time()
        try:
            AllChem.EmbedMultipleConfs(mol, numConfs, numThreads=num_cpus, useRandomCoords=True)
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=num_cpus, maxIters=max_ff_iters)
        except:
            print(f"{mol_name} not working - in time: {time.time() - t}") 
            continue
        
        # Cluster Conformers
        mol = butina_clustering_m(mol, threshold=0.01)
        print(f"{mol_name} - Clustering - before: {numConfs}, after {mol.GetNumConformers()}, time: {time.time() - t}")
        
        # Compute FF energy
        mm_data_form = namedtuple("MM_Data", 'idx energy')
        mm_data = []
        for i, conf in enumerate(mol.GetConformers()):
            tm = Chem.Mol(mol,False,conf.GetId())
            
            # UFF
            ff = AllChem.UFFGetMoleculeForceField(tm)
            
            # MMFF
            #prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant="MMFF94")
            #ff = AllChem.MMFFGetMoleculeForceField(tm,prop) 
            
            mm_data.append(mm_data_form(conf.GetId(), ff.CalcEnergy()))
        
        mm_data = pd.DataFrame(mm_data)
        
        # Use X low energy confs.
        min_idx = mm_data.nsmallest(5, 'energy')['idx'].values
        
        letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
        for i, conf_idx in enumerate(min_idx):
            sdf = Chem.MolToMolBlock(mol, confId=int(conf_idx))
            rdkit_mol = Chem.MolFromMolBlock(sdf, removeHs=False)
            
            # Sample linker for now just rotate one end
            # the cages are symmetric
            rdkit_mol = sample_contact(rdkit_mol, 120.)

            # opt after sampling contact.
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=num_cpus, maxIters=max_ff_iters)

            # write mols
            write_rdkit_confs(rdkit_mol, mol_name + '-' + letters[i])        




















        
        #obConversion = openbabel.OBConversion()
        #obConversion.SetInAndOutFormats("sdf", "xyz")

        #xyz = ''
        #for conf_idx in min_idx:
        #    new_conf = Chem.MolToMolBlock(mol, confId=int(conf_idx))
        #    newConfm = openbabel.OBMol()
        #    obConversion.ReadString(newConfm, new_conf)

        #    xyz += obConversion.WriteString(newConfm)

        #with open('test.xyz', 'w') as w:
        #    w.write(xyz)

            #name = csv.loc[0].comp_name + f'-{i}'
            #Chem.MolToMolFile(mol, name + '.sdf')
            #names.append(name)
        

        # This part computed the GFN2-xTB
        #with Pool(num_cpus) as pool:
        #    c = pool.map(get_xtb_opt_conf, names)#

        #qmmol = make_qmmol(c)
        #best = qmmol.nlowest(1)[0]
        
        
        # Right now it is not the connectivity might not be correct
        #best = AC2mol(best.get_rdkit_mol(), orgAC, best.atomic_numbers, 0, False, True)
        #Chem.MolToMolFile(best.get_rdkit_mol(), 'test.sdf')


        # save conf
        #dat = [(csv.loc[0].comp_name, best, best.results['energy'])]
        #data = pd.DataFrame(dat, columns=['comp_name', 'best_conf', 'energy'])
        #
        #data.to_pickle(csv.loc[0].comp_name + '.pkl')
    

