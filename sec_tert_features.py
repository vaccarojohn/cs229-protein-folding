import os
import requests
import tempfile
import numpy as np
import pandas as pd
import pydssp
from biotite.structure.io import load_structure
from biotite.structure.celllist import CellList
from biotite.structure import get_residue_positions

def get_sec_tert_structure(seq):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=seq)
    
    alpha_carbons = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, 'temp.pdb')

        text = response.content.decode('utf-8')
        with open(temp_file_path, 'w') as f:
            f.write(text)

        sec_coord = pydssp.read_pdbtext(text, return_sequence=False)
        dssp = ''.join(map(str, pydssp.assign(sec_coord, out_type='index')))
        atom_array = load_structure(temp_file_path, extra_fields=['b_factor'])
        alpha_carbons = atom_array[atom_array.atom_name == 'CA']

    return (alpha_carbons, dssp)

def main():
    df = pd.read_csv('data_with_subsequence_data.csv')
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'

    for index, row in df.iterrows():
        if not normal_mask[index]:
            df.loc[index, 'CO'] = 0
            df.loc[index, 'Abs_CO'] = 0
            df.loc[index, 'TCD'] = 0
            df.loc[index, 'LR_CO'] = 0
            df.loc[index, 'B_factor'] = 0
            df.loc[index, 'DSSP'] = ''
            continue

        print(row['Sub-Sequence Used In Experiment'])
        alpha_carbons, dssp = get_sec_tert_structure(row['Sub-Sequence Used In Experiment'])

        df.loc[index, 'DSSP'] = dssp
        df.loc[index, 'B_factor'] = round(np.mean(alpha_carbons.b_factor), 4)

        cell_list = CellList(alpha_carbons, cell_size=5)
        contact_atom_indices_six = cell_list.get_atoms(alpha_carbons.coord, radius=6)
        contact_atom_indices_eight = cell_list.get_atoms(alpha_carbons.coord, radius=8)

        l = len(alpha_carbons)
        sep_co = 0
        sep_lrco = 0
        n_co = 0
        n_lrco = 0

        for i in range(contact_atom_indices_six.shape[0]):
            atom_mask = contact_atom_indices_six[i] != -1
            delta_s = np.abs(get_residue_positions(alpha_carbons, contact_atom_indices_six[i][atom_mask]) - i)

            far_mask = delta_s >= 3
            sep_co += np.sum(delta_s[far_mask])
            n_co += np.sum(far_mask)

        for i in range(contact_atom_indices_eight.shape[0]):
            atom_mask = contact_atom_indices_eight[i] != -1
            delta_s = np.abs(get_residue_positions(alpha_carbons, contact_atom_indices_eight[i][atom_mask]) - i)

            far_mask = delta_s >= 12
            sep_lrco += np.sum(delta_s[far_mask])
            n_lrco += np.sum(far_mask)

        if n_co == 0:
            df.loc[index, 'CO'] = 0
            df.loc[index, 'Abs_CO'] = 0
        else:
            df.loc[index, 'CO'] = round(sep_co / (n_co * l), 4)
            df.loc[index, 'Abs_CO'] = round(sep_co / n_co, 4)

        df.loc[index, 'TCD'] = round(sep_co / (l * l), 4)
        
        if n_lrco == 0:
            df.loc[index, 'LR_CO'] = 0
        else:
            df.loc[index, 'LR_CO'] = round(sep_lrco / (n_lrco * l), 4)

    df.to_csv('data_with_sec_tert_structure.csv', index=False)

if __name__ == "__main__":
    main()
