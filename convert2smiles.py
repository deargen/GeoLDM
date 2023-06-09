import os
import pickle
from rdkit import Chem
import torch
from mol_gen.models.GeoLDM.configs.datasets_config import get_dataset_info
from mol_gen.models.GeoLDM.qm9.visualizer import load_molecule_xyz
from mol_gen.models.GeoLDM.qm9.rdkit_functions import build_molecule


def convert2smiles(model_path, output_dir):
    with open(os.path.join(model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    target_dir = output_dir + "/generated_molecules/"
    
    gen_mols = []
    for file in os.listdir(target_dir):
        tmp = target_dir + file
        positions, one_hot, charges = load_molecule_xyz(tmp, dataset_info)

        one_hot = torch.argmax(one_hot, dim=-1)
        atom_type = one_hot.squeeze(0).cpu().detach().numpy()
        mol = build_molecule(positions, atom_type, dataset_info)
        gen_mols.append(Chem.MolToSmiles(mol))
        
    with open(output_dir + "/SMILES.txt", 'w') as f:
        for smi in gen_mols:
            f.write("%s\n" % smi)